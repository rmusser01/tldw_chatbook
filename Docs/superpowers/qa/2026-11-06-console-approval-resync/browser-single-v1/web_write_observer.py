"""Private diagnostic only: read-only fd state around one original write call."""
from collections import deque
from functools import partial
import ctypes as C
from ctypes import wintypes as W
import json
import msvcrt
import os
import sys
import threading
import time
import traceback


class WriteObservation:
    def __init__(self, output):
        self.output=output;self.startup=[];self.recent=deque(maxlen=64)
        self.count=0;self.failures=0;self.errors=[];self.error_count=0
        self.binding=None;self.driver=None;self.original=None;self.wrapper=None
        self.lock=threading.Lock()
        self.kernel=C.WinDLL('kernel32',use_last_error=True)
        self.kernel.GetFileType.argtypes=[W.HANDLE];self.kernel.GetFileType.restype=W.DWORD
        self.kernel.GetNamedPipeHandleStateW.argtypes=[W.HANDLE,C.POINTER(W.DWORD),C.POINTER(W.DWORD),C.POINTER(W.DWORD),C.POINTER(W.DWORD),W.LPWSTR,W.DWORD]
        self.kernel.GetNamedPipeHandleStateW.restype=W.BOOL

    def _observer_error(self, phase, error):
        try:
            with self.lock:
                self.error_count+=1
                if len(self.errors)<8:self.errors.append({'phase':phase,'type':type(error).__name__})
        except BaseException:
            pass  # Observer failure must never replace original I/O behavior.

    def _descriptor_state(self, fd):
        state={}
        try:state['blocking']=os.get_blocking(fd)
        except BaseException as error:state['blocking_error_type']=type(error).__name__
        try:
            handle=msvcrt.get_osfhandle(fd);state['os_handle']=handle
            C.set_last_error(0);kind=int(self.kernel.GetFileType(W.HANDLE(handle)))
            state.update(file_type=kind,file_type_winerror=C.get_last_error() if kind==0 else 0)
            if kind==3:
                flags=W.DWORD();C.set_last_error(0)
                success=bool(self.kernel.GetNamedPipeHandleStateW(W.HANDLE(handle),C.byref(flags),None,None,None,None,0))
                state.update(pipe_state_success=success,pipe_state=flags.value if success else None,pipe_state_winerror=0 if success else C.get_last_error())
        except BaseException as error:state['handle_state_error_type']=type(error).__name__
        return state

    def _keep(self, row):
        with self.lock:
            if row['sequence']<=4:self.startup.append(row)
            self.recent.append(row)

    def _call(self, original, fd, payload):
        row=None
        try:
            with self.lock:self.count+=1;sequence=self.count
            row={'sequence':sequence,'fd':fd,'attempted_bytes':len(payload),'before_ns':time.perf_counter_ns(),**self._descriptor_state(fd)}
        except BaseException as error:self._observer_error('before_write',error)
        try:
            returned=original(payload)  # EXACT original target/payload, once.
        except BaseException as original_error:
            try:
                with self.lock:self.failures+=1
                if row is not None:
                    errno=getattr(original_error,'errno',None);winerror=getattr(original_error,'winerror',None)
                    row.update(after_ns=time.perf_counter_ns(),outcome='raised',error_type=type(original_error).__name__,errno=errno if type(errno) is int else None,winerror=winerror if type(winerror) is int else None,code_frames=[{'file':frame.filename,'function':frame.name,'line':frame.lineno} for frame in traceback.extract_tb(original_error.__traceback__)])
                    self._keep(row)
                self.persist()
            except BaseException as error:self._observer_error('failed_write_metadata',error)
            raise  # Preserve the ORIGINAL exception; no retry or write fallback.
        else:
            try:
                if row is not None:
                    row.update(after_ns=time.perf_counter_ns(),outcome='returned',return_count=returned if type(returned) is int else None)
                    self._keep(row)
            except BaseException as error:self._observer_error('successful_write_metadata',error)
            return returned  # Preserve original return object/count unchanged.

    def bind(self, driver, app):
        try:
            from textual.drivers.web_driver import WebDriver
            fd=driver.fileno;original=driver._write
            if self.driver is not None:raise RuntimeError('Observer_already_bound')
            if type(driver) is not WebDriver or driver._app is not app:raise RuntimeError('Not_exact_owned_stock_WebDriver')
            if type(fd) is not int or fd<0 or fd!=sys.__stdout__.fileno():raise RuntimeError('Stock_stdout_fd_mismatch')
            if not isinstance(original,partial) or original.func is not os.write or original.args!=(fd,) or original.keywords:raise RuntimeError('Stock_original_os_write_partial_mismatch')
            def observed(payload, /):return self._call(original,fd,payload)
            self.binding={'driver_module':type(driver).__module__,'driver_class':type(driver).__name__,'fd':fd,'exact_os_write_partial':True,'same_owned_stdout_fd':True,'no_global_os_write_patch':True}
            self.driver=driver;self.original=original;self.wrapper=observed
            driver._write=observed
        except BaseException as error:self._observer_error('bind',error)
        return driver

    def snapshot(self):
        with self.lock:
            return {'binding':self.binding,'attempts':self.count,'failures':self.failures,'startup':list(self.startup),'recent':list(self.recent),'observer_errors':list(self.errors),'observer_error_count':self.error_count,'records_bound':{'startup':4,'recent':64,'observer_errors':8},'payload_content_recorded':False,'mode_changes':False,'write_retries':False,'diagnostic_only':True,'qualified':False,'product_samples':0}

    def persist(self):
        try:self.output.write_text(json.dumps(self.snapshot(),indent=2)+'\n',encoding='utf-8')
        except BaseException as error:self._observer_error('persist',error)
