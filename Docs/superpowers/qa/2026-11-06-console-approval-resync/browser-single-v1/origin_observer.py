"""Bounded metadata-only origin diagnostic; never changes original execution."""
from collections import deque
from contextlib import contextmanager
import json
import math
import re
import threading
import time
import traceback
from urllib.parse import urlsplit


class OriginObservation:
    KINDS={'child_info','page_mainframe_navigation','page_crash','page_close','context_close','browser_disconnected','ws_owned','ws_closed','ws_close_frame','main_error_before_cleanup','cleanup_start','app_exit','write_meta'}
    OPERATIONS={'control_wait','control_read_parse','source_assert','process_object','exe_resolve_assert','parents_assert','net_connections','listener_assert','guard_read_parse','guard_start_assert','guard_executable_assert','guard_prefix_assert','allow_control_port','cache_identity','identity_receipt_write'}

    def __init__(self, output):
        self.output=output;self.lock=threading.Lock();self.startup=[];self.recent=deque(maxlen=64)
        self.count=0;self.errors=[];self.error_count=0;self.first={};self.ws_ids=set()
        self.app_bound=False;self.meta_bound=False;self.browser_bound=False;self.expected_port=None

    def observer_error(self, phase, error):
        try:
            with self.lock:
                self.error_count+=1
                if len(self.errors)<8:self.errors.append({'phase':phase,'type':type(error).__name__})
        except BaseException:pass

    @staticmethod
    def error_metadata(error):
        errno=getattr(error,'errno',None);winerror=getattr(error,'winerror',None)
        return {'type':type(error).__name__,'errno':errno if type(errno) is int else None,'winerror':winerror if type(winerror) is int else None,'code_frames':[{'file':f.filename,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(error.__traceback__,limit=32)]}

    def record(self,kind,*,phase=None,operation=None,direction=None,request_id=None,cdp_timestamp=None,close_code=None,meta_type=None,new_tab=None,code_frames=None,error=None):
        try:
            if kind not in self.KINDS:raise ValueError('Unknown_observation_kind')
            row={'timestamp_ns':time.perf_counter_ns(),'kind':kind}
            if phase is not None:
                if phase not in {'entry','return','raise'}:raise ValueError('Unknown_phase')
                row['phase']=phase
            if operation is not None:
                if operation not in self.OPERATIONS:raise ValueError('Unknown_child_info_operation')
                row['operation']=operation
            if direction is not None:
                if direction not in {'sent','received'}:raise ValueError('Unknown_close_direction')
                row['direction']=direction
            if request_id is not None:
                if type(request_id) is not str or re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',request_id) is None:raise ValueError('Invalid_request_identifier')
                row['request_id']=request_id
            if cdp_timestamp is not None:
                if type(cdp_timestamp) not in {int,float} or not math.isfinite(cdp_timestamp):raise ValueError('Invalid_CDP_timestamp')
                row['cdp_timestamp']=cdp_timestamp
            if kind=='ws_close_frame':row['opcode']=8;row['close_code']=close_code
            if close_code is not None and (type(close_code) is not int or not 0<=close_code<=65535):raise ValueError('Invalid_close_code')
            if meta_type is not None:
                if meta_type not in {'exit','open_url','other'}:raise ValueError('Unknown_meta_type')
                row['meta_type']=meta_type
            if type(new_tab) is bool:row['new_tab']=new_tab
            if code_frames is not None:row['code_frames']=code_frames
            if error is not None:row['error']=self.error_metadata(error)
            important=kind in {'ws_close_frame','ws_closed','main_error_before_cleanup','cleanup_start','app_exit','page_crash','page_close','context_close','browser_disconnected'} or (kind=='write_meta' and meta_type in {'exit','open_url'})
            with self.lock:
                self.count+=1;row['sequence']=self.count
                if self.count<=4:self.startup.append(row)
                self.recent.append(row)
                first_key=kind+('_'+direction if direction else '')
                if important and first_key not in self.first and len(self.first)<16:self.first[first_key]=row
            if important or error is not None:self.persist()
        except BaseException as error:self.observer_error('record',error)

    @contextmanager
    def span(self,operation):
        self.record('child_info',phase='entry',operation=operation)
        try:yield
        except BaseException as original_error:
            self.record('child_info',phase='raise',operation=operation,error=original_error)
            raise
        else:self.record('child_info',phase='return',operation=operation)

    def delegate(self,original,kind,args,kwargs,fields):
        self.record(kind,phase='entry',**fields)
        try:returned=original(*args,**kwargs)
        except BaseException as original_error:
            self.record(kind,phase='raise',error=original_error,**fields)
            raise
        else:
            self.record(kind,phase='return',**fields)
            return returned

    def bind_app(self,app,expected_exit):
        try:
            original=app.exit
            if self.app_bound or getattr(original,'__self__',None) is not app or getattr(original,'__func__',None) is not expected_exit:raise RuntimeError('Not_original_owned_App_exit')
            def observed(*args,**kwargs):
                fields={}
                try:fields['code_frames']=[{'file':f.filename,'function':f.name,'line':f.lineno} for f in traceback.extract_stack(limit=33)[:-1]]
                except BaseException as error:self.observer_error('app_exit_stack',error)
                return self.delegate(original,'app_exit',args,kwargs,fields)
            app.exit=observed;self.app_bound=True
        except BaseException as error:self.observer_error('bind_app',error)

    def bind_meta(self,driver,app,expected_class):
        try:
            original=driver.write_meta
            if self.meta_bound or type(driver) is not expected_class or driver._app is not app or getattr(original,'__self__',None) is not driver or getattr(original,'__func__',None) is not expected_class.write_meta:raise RuntimeError('Not_original_owned_WebDriver_write_meta')
            def observed(*args,**kwargs):
                fields={'meta_type':'other'}
                try:
                    data=args[0] if args else kwargs.get('data')
                    if type(data) is dict:
                        kind=data.get('type')
                        fields['meta_type']=kind if type(kind) is str and kind in {'exit','open_url'} else 'other'
                        if type(data.get('new_tab')) is bool:fields['new_tab']=data['new_tab']
                except BaseException as error:self.observer_error('write_meta_fields',error)
                return self.delegate(original,'write_meta',args,kwargs,fields)
            driver.write_meta=observed;self.meta_bound=True
        except BaseException as error:self.observer_error('bind_meta',error)

    def ws_created(self,event):
        try:
            target=urlsplit(event['url']);request_id=event['requestId']
            if target.scheme!='ws' or target.hostname!='127.0.0.1' or target.port!=self.expected_port or target.path!='/ws' or target.username is not None or target.password is not None:return
            if type(request_id) is not str or re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}',request_id) is None:raise ValueError('Invalid_owned_request_identifier')
            if len(self.ws_ids)>=8 and request_id not in self.ws_ids:raise ValueError('Owned_WS_identifier_bound')
            self.ws_ids.add(request_id)
            self.record('ws_owned',request_id=request_id)
        except BaseException as error:self.observer_error('ws_created',error)

    def ws_frame(self,direction,event):
        try:
            request_id=event.get('requestId')
            if request_id not in self.ws_ids:return
            response=event.get('response',{})
            opcode=response.get('opcode')
            if type(opcode) not in {int,float} or opcode!=8:return  # Never inspect other payloads.
            payload=response.get('payloadData')
            if type(payload) is not str:raise ValueError('Missing_base64_close_frame')
            code=None
            if payload:
                if len(payload)<4 or len(payload)>168 or len(payload)%4:raise ValueError('Malformed_close_frame_size')
                # Decode ONLY the two-byte close code from the first three base64
                # characters. Never decode/read/persist the remaining reason.
                alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
                a,b,c=(alphabet.index(char) for char in payload[:3])
                code=(((a<<2)|(b>>4))<<8)|(((b&15)<<4)|(c>>2))
            self.record('ws_close_frame',direction=direction,request_id=request_id,cdp_timestamp=event['timestamp'],close_code=code)
        except BaseException as error:self.observer_error('ws_frame',error)

    def ws_closed(self,event):
        try:
            if event.get('requestId') in self.ws_ids:self.record('ws_closed',request_id=event['requestId'],cdp_timestamp=event['timestamp'])
        except BaseException as error:self.observer_error('ws_closed',error)

    async def bind_browser(self,browser,context,page,port):
        try:
            if self.browser_bound or type(port) is not int:raise RuntimeError('Invalid_browser_origin_binding')
            self.expected_port=port
            def navigated(frame):
                try:
                    if frame is page.main_frame:self.record('page_mainframe_navigation')
                except BaseException as error:self.observer_error('navigation',error)
            page.on('framenavigated',navigated)
            page.on('crash',lambda *_:self.record('page_crash'))
            page.on('close',lambda *_:self.record('page_close'))
            context.on('close',lambda *_:self.record('context_close'))
            browser.on('disconnected',lambda *_:self.record('browser_disconnected'))
            cdp=await context.new_cdp_session(page)
            cdp.on('Network.webSocketCreated',self.ws_created)
            cdp.on('Network.webSocketFrameSent',lambda event:self.ws_frame('sent',event))
            cdp.on('Network.webSocketFrameReceived',lambda event:self.ws_frame('received',event))
            cdp.on('Network.webSocketClosed',self.ws_closed)
            await cdp.send('Network.enable')  # Observation only, before original goto.
            self.cdp=cdp;self.browser_bound=True
        except BaseException as error:self.observer_error('bind_browser',error)

    def snapshot(self):
        with self.lock:
            return {'events':self.count,'startup':list(self.startup),'recent':list(self.recent),'first':dict(self.first),'observer_errors':list(self.errors),'observer_error_count':self.error_count,'owned_ws_count':len(self.ws_ids),'app_bound':self.app_bound,'meta_bound':self.meta_bound,'browser_bound':self.browser_bound,'record_bounds':{'startup':4,'recent':64,'first_kinds':16,'errors':8,'code_frames':32,'owned_ws_ids':8},'body_url_reason_or_args_recorded':False,'execution_protocol_or_timeout_changes':False,'diagnostic_only':True,'qualified':False,'product_samples':0}

    def persist(self):
        try:self.output.write_text(json.dumps(self.snapshot(),indent=2)+'\n',encoding='utf-8')
        except BaseException as error:self.observer_error('persist',error)
