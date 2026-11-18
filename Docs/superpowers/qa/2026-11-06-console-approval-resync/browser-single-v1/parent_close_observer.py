"""Private diagnostic: transparent owned-server/service close and error events."""
import asyncio
from collections import deque
import json
import threading
import time
import traceback


class ParentCloseObservation:
    def __init__(self, output):
        self.output=output;self.lock=threading.Lock();self.startup=[];self.recent=deque(maxlen=64)
        self.count=0;self.errors=[];self.error_count=0;self.first_error=None;self.first_close=None
        self.services=0;self.server_bound=False

    def _observer_error(self, phase, error):
        try:
            with self.lock:
                self.error_count+=1
                if len(self.errors)<8:self.errors.append({'phase':phase,'type':type(error).__name__})
        except BaseException:pass

    @staticmethod
    def _error(error):
        errno=getattr(error,'errno',None);winerror=getattr(error,'winerror',None)
        return {'type':type(error).__name__,'errno':errno if type(errno) is int else None,'winerror':winerror if type(winerror) is int else None,'code_frames':[{'file':frame.filename,'function':frame.name,'line':frame.lineno} for frame in traceback.extract_tb(error.__traceback__)]}

    def _ws_state(self, websocket):
        if websocket is None:return {'available':False}
        code=websocket.close_code;error=websocket.exception()
        return {'available':True,'closed':bool(websocket.closed),'close_code':int(code) if isinstance(code,int) and not isinstance(code,bool) else None,'exception':self._error(error) if error is not None else None}

    def _record(self, kind, phase, websocket=None, error=None, **metadata):
        try:
            row={'timestamp_ns':time.perf_counter_ns(),'kind':kind,'phase':phase,'websocket':self._ws_state(websocket),**metadata}
            if error is not None:row['error']=self._error(error)
            with self.lock:
                self.count+=1;row['sequence']=self.count
                if self.count<=4:self.startup.append(row)
                self.recent.append(row)
                first_error=error is not None and self.first_error is None
                first_close=self.first_close is None and (row['websocket'].get('closed') is True or (kind in {'remote_close','stop'} and phase=='entry') or (kind=='authenticated_messages' and phase=='return'))
                if first_error:self.first_error=row
                if first_close:self.first_close=row
            if first_error or first_close:self.persist()
        except BaseException as observer_error:self._observer_error('record',observer_error)

    async def _delegate(self, original, kind, websocket, args, kwargs, metadata=None):
        fields={}
        try:fields=metadata() if metadata is not None else {}
        except BaseException as error:self._observer_error('delegate_metadata',error)
        self._record(kind,'entry',websocket,**fields)
        try:returned=await original(*args,**kwargs)  # Exact target/objects once.
        except BaseException as original_error:
            self._record(kind,'raise',websocket,error=original_error,**fields)
            raise
        else:
            self._record(kind,'return',websocket,**fields)
            return returned

    def _wrap(self, original, kind, websocket, length=False):
        async def observed(*args,**kwargs):
            metadata=(lambda:{'attempted_bytes':len(args[0])}) if length and args else None
            return await self._delegate(original,kind,websocket,args,kwargs,metadata)
        return observed

    async def _run(self, service, original, websocket, args, kwargs):
        reader=None;original_read=None;observed_read=None;missing=object();prior_local=missing
        try:
            reader=service._process.stdout
            original_read=reader.readexactly
            prior_local=reader.__dict__.get('readexactly',missing)
            if getattr(original_read,'__self__',None) is not reader or getattr(original_read,'__func__',None) is not asyncio.StreamReader.readexactly:
                raise RuntimeError('Not_stock_owned_StreamReader_readexactly')
            packet_phase=0
            async def observed_read(n):
                nonlocal packet_phase
                label=('type','length','payload')[packet_phase]
                self._record('readexactly','entry',websocket,requested_bytes=n,packet_phase=label)
                try:data=await original_read(n)
                except BaseException as original_error:
                    fields={'requested_bytes':n,'packet_phase':label}
                    try:
                        if isinstance(original_error,asyncio.IncompleteReadError):fields['partial_bytes']=len(original_error.partial)
                    except BaseException as error:self._observer_error('partial_length',error)
                    self._record('readexactly','raise',websocket,error=original_error,**fields)
                    raise
                else:
                    try:
                        fields={'requested_bytes':n,'returned_bytes':len(data),'packet_phase':label}
                        if label=='type':fields['packet_tag']={b'D':'D',b'M':'M',b'P':'P'}.get(data,'other')
                        packet_phase=(packet_phase+1)%3
                        self._record('readexactly','return',websocket,**fields)
                    except BaseException as error:self._observer_error('read_metadata',error)
                    return data  # Exact original bytes; no copying/modification.
            reader.readexactly=observed_read  # Per-instance, before run captures it.
        except BaseException as error:self._observer_error('read_bind',error)
        try:return await self._delegate(original,'service_run',websocket,args,kwargs)
        finally:
            try:
                if reader is not None and observed_read is not None and getattr(reader,'readexactly',None) is observed_read:
                    if prior_local is missing:del reader.readexactly
                    else:reader.readexactly=prior_local  # Exact prior instance state.
            except BaseException as error:self._observer_error('read_restore',error)

    def bind_service(self, service, original_class):
        try:
            if type(service) is not original_class:raise RuntimeError('Actual_service_class_changed')
            websocket=getattr(service.remote_write_bytes,'__self__',None)
            if websocket is None or getattr(service.remote_close,'__self__',None) is not websocket:raise RuntimeError('Owned_websocket_callback_identity_mismatch')
            missing=object();names=('remote_write_bytes','remote_close','stop','run','_qa_parent_close_websocket')
            prior={name:service.__dict__.get(name,missing) for name in names};installed={}
            def restore(name):
                try:
                    current=installed.pop(name,missing)
                    if current is not missing and service.__dict__.get(name,missing) is current:
                        if prior[name] is missing:delattr(service,name)
                        else:setattr(service,name,prior[name])
                except BaseException as error:self._observer_error('service_restore',error)
            service._qa_parent_close_websocket=websocket;installed['_qa_parent_close_websocket']=websocket
            with self.lock:self.services+=1;index=self.services
            self._record('service_bound','entry',websocket,service_index=index)
            for name in ('remote_write_bytes','remote_close'):
                original=getattr(service,name)
                wrapped=self._wrap(original,name,websocket,length=name=='remote_write_bytes')
                setattr(service,name,wrapped);installed[name]=wrapped
            original_run=service.run;original_stop=service.stop
            async def run(*args,**kwargs):
                try:return await self._run(service,original_run,websocket,args,kwargs)
                finally:restore('run')
            async def stop(*args,**kwargs):
                try:return await self._delegate(original_stop,'stop',websocket,args,kwargs)
                finally:
                    # Remove observer-bound method cycles when original stop
                    # retires, preserving original service/pipe teardown lifetime.
                    for name in tuple(installed):restore(name)
            service.run=run;installed['run']=run
            service.stop=stop;installed['stop']=stop
        except BaseException as error:self._observer_error('service_bind',error)
        return service

    def bind_server(self, server):
        original_factory=server._chatbook_app_service_class
        original_process=server._process_authenticated_messages
        if not isinstance(original_factory,type) or not callable(original_process):
            self._observer_error('server_bind',RuntimeError('Actual_server_seam_missing'))
            return server  # Leave original missing-seam behavior untouched.
        def factory(*args,**kwargs):
            try:actual=original_factory(*args,**kwargs)  # Original constructor once.
            except BaseException as original_error:
                self._record('service_factory','raise',error=original_error)
                raise
            self.bind_service(actual,original_factory)
            return actual
        async def process(*args,**kwargs):
            websocket=args[0] if args else kwargs.get('websocket')
            service=args[1] if len(args)>1 else kwargs.get('app_service')
            verified=getattr(service,'_qa_parent_close_websocket',None) is websocket and websocket is not None
            if not verified:
                self._observer_error('websocket_identity',RuntimeError('Owned_websocket_identity_missing'))
                websocket=None  # Missing observation, never pretend ownership.
            return await self._delegate(original_process,'authenticated_messages',websocket,args,kwargs)
        server._chatbook_app_service_class=factory
        server._process_authenticated_messages=process
        self.server_bound=True

    def snapshot(self):
        with self.lock:return {'events':self.count,'startup':list(self.startup),'recent':list(self.recent),'first_original_error':self.first_error,'first_close_observation':self.first_close,'observer_errors':list(self.errors),'observer_error_count':self.error_count,'services':self.services,'server_bound':self.server_bound,'record_bounds':{'startup':4,'recent':64,'first_error':1,'first_close':1,'observer_errors':8},'content_recorded':False,'modes_protocol_heartbeat_or_timeouts_changed':False,'diagnostic_only':True,'qualified':False,'product_samples':0}

    def persist(self):
        try:self.output.write_text(json.dumps(self.snapshot(),indent=2)+'\n',encoding='utf-8')
        except BaseException as error:self._observer_error('persist',error)
