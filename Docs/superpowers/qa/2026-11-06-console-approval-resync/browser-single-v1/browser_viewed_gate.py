"""Owned headed Edge visibility gate. Read-only native state and owned pixels only.

Checks bracket each diagnostic; they do not prove continuous nonocclusion or
physical scanout. CDP presentation joins remain a separate required observer.
"""
import asyncio
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from native_window_api import win32, geometry
from owned_roi_capture import capture_owned_roi


class ViewedBrowserGate:
    @classmethod
    async def bind(cls, browser, page, run, executable):
        import psutil
        cdp=await browser.new_browser_cdp_session()
        try:
            infos=(await cdp.send('SystemInfo.getProcessInfo'))['processInfo']
            args=(await cdp.send('Browser.getBrowserCommandLine'))['arguments']
        finally:await cdp.detach()
        pids=[int(row['id']) for row in infos if row['type']=='browser']
        if len(pids)!=1:raise RuntimeError('Browser_process_identity_ambiguous')
        profiles=[Path(arg.split('=',1)[1]).resolve() for arg in args if arg.startswith('--user-data-dir=')]
        if len(profiles)!=1 or not profiles[0].is_relative_to((run/'product-profile/temp').resolve()):
            raise RuntimeError('Browser_profile_not_owned')
        process=psutil.Process(pids[0])
        if Path(process.exe()).resolve()!=executable.resolve() or os.getpid() not in {p.pid for p in process.parents()}:
            raise RuntimeError('Browser_native_identity_mismatch')
        self=cls();self.pid=pids[0];self.page=page;self.run=run
        self.user,self.kernel,self.gdi=win32()
        enumproc=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
        self.user.EnumWindows.argtypes=[enumproc,W.LPARAM];self.user.EnumWindows.restype=W.BOOL
        candidates=[]
        @enumproc
        def visit(hwnd,_):
            pid=W.DWORD();self.user.GetWindowThreadProcessId(hwnd,C.byref(pid))
            if pid.value==self.pid and self.user.IsWindowVisible(hwnd):
                name=C.create_unicode_buffer(128);self.user.GetClassNameW(hwnd,name,128)
                if name.value=='Chrome_WidgetWin_1':candidates.append(int(hwnd))
            return True
        if not self.user.EnumWindows(visit,0) or len(candidates)!=1:raise RuntimeError('Owned_visible_HWND_ambiguous')
        self.hwnd=candidates[0];self.expected=geometry(self.user,self.hwnd)
        self.identity={'browser_pid':self.pid,'hwnd':self.hwnd,'private_profile_verified':True,'descendant_verified':True,'geometry':self.expected}
        return self

    def os_state(self):
        actual=geometry(self.user,self.hwnd)
        return {'timestamp_ns':time.perf_counter_ns(),'hwnd':self.hwnd,'owned_pid_matches':actual['host_pid']==self.pid,'visible':bool(self.user.IsWindowVisible(self.hwnd)),'iconic':bool(self.user.IsIconic(self.hwnd)),'foreground_matches':self.user.GetAncestor(self.user.GetForegroundWindow(),2)==self.hwnd,'geometry':actual}

    async def activate_owned(self,observations):
        # Actual activation only; no foreground/visibility assertion is waived.
        initial=self.os_state()
        if not initial['owned_pid_matches'] or initial['geometry']['class']!='Chrome_WidgetWin_1':
            raise RuntimeError('Owned_browser_activation_identity_mismatch')
        for _ in range(5):
            await asyncio.wait_for(self.page.bring_to_front(),2)
            state=self.os_state();observations.append(state)
            if state['visible'] and not state['iconic'] and state['foreground_matches']:break
            await asyncio.sleep(.1)
        else:
            state=self.os_state()
            if not state['owned_pid_matches'] or state['geometry']['class']!='Chrome_WidgetWin_1':
                raise RuntimeError('Owned_browser_activation_identity_mismatch')
            if state['iconic'] or not state['visible']:self.user.ShowWindow(self.hwnd,9)
            self.user.SetForegroundWindow.argtypes=[W.HWND];self.user.SetForegroundWindow.restype=W.BOOL
            observations.append({'native_set_foreground_returned':bool(self.user.SetForegroundWindow(self.hwnd)),'timestamp_ns':time.perf_counter_ns()})
            for _ in range(20):
                await asyncio.sleep(.05);state=self.os_state();observations.append(state)
                if state['visible'] and not state['iconic'] and state['foreground_matches']:break
            else:
                observations.append({'verified_caption_activation':self._click_verified_caption(),'timestamp_ns':time.perf_counter_ns()})
                for _ in range(20):
                    await asyncio.sleep(.05);state=self.os_state();observations.append(state)
                    if state['visible'] and not state['iconic'] and state['foreground_matches']:break
                else:raise RuntimeError('Owned_browser_caption_click_did_not_activate')
        # Stabilize actual geometry BEFORE any observation/qualification begins.
        # Both initial and final native/page geometry remain in the receipt.
        self.expected=state['geometry'];self.identity['geometry']=self.expected
        return {'initial_state':initial,'final_state':state,'page_viewport':await self.page.evaluate('({inner:[innerWidth,innerHeight],outer:[outerWidth,outerHeight],devicePixelRatio})'),'foreground_gate_waived':False}

    def _click_verified_caption(self):
        """One real caption click, exclusively on this verified owned HWND.

        No page-control coordinates, generic input, Alt tricks or TOPMOST.
        Search and immediate recheck require actual WM_NCHITTEST==HTCAPTION.
        """
        u=self.user;h=self.hwnd
        attempt=time.perf_counter_ns()
        def record(phase,**facts):
            row={"phase":phase,"attempt_ns":attempt,"observed_ns":time.perf_counter_ns(),"hwnd":h,"approval_action":False,**facts}
            with (self.run/"browser-caption-activation.jsonl").open("a",encoding="utf-8") as output:output.write(json.dumps(row)+"\n")
        def own():
            state=self.os_state()
            if not state['owned_pid_matches'] or state['geometry']['class']!='Chrome_WidgetWin_1':
                raise RuntimeError('Owned_browser_caption_identity_mismatch')
            return state
        own();u.ShowWindow(h,9)  # Normal restore of OWN window only.
        for name,restype,args in [
            ('SetWindowPos',W.BOOL,[W.HWND,W.HWND,C.c_int,C.c_int,C.c_int,C.c_int,W.UINT]),
            ('GetWindowRect',W.BOOL,[W.HWND,C.POINTER(W.RECT)]),
            ('GetTopWindow',W.HWND,[W.HWND]),('GetWindow',W.HWND,[W.HWND,W.UINT]),
            ('GetSystemMetrics',C.c_int,[C.c_int]),
            ('GetCursorPos',W.BOOL,[C.POINTER(W.POINT)]),
            ('GetAsyncKeyState',W.SHORT,[C.c_int]),
            ('SendMessageTimeoutW',W.LPARAM,[W.HWND,W.UINT,W.WPARAM,W.LPARAM,W.UINT,W.UINT,C.POINTER(C.c_size_t)]),
        ]:
            function=getattr(u,name);function.restype=restype;function.argtypes=args
        own()
        # HWND_TOP0 is ordinary Z-order; the NOACTIVATE flag leaves actual
        # foreground to the subsequent normal caption gesture. Never TOPMOST.
        if not u.SetWindowPos(h,W.HWND(0),0,0,0,0,0x0001|0x0002|0x0010):
            raise RuntimeError('Owned_browser_normal_raise_refused')
        bounds=W.RECT()
        if not u.GetWindowRect(h,C.byref(bounds)):raise RuntimeError('Owned_browser_caption_geometry_unavailable')
        vx,vy,vw,vh=[u.GetSystemMetrics(index) for index in (76,77,78,79)]
        if vw<=1 or vh<=1:raise RuntimeError('Owned_browser_virtual_screen_unavailable')
        def point_is_caption(x,y):
            state=own()
            if not state['visible'] or state['iconic'] or not (vx<=x<vx+vw and vy<=y<vy+vh):return False
            if u.GetAncestor(u.WindowFromPoint(W.POINT(x,y)),2)!=h:return False
            window=u.GetTopWindow(None)
            for _ in range(1024):
                if not window:return False
                if int(window)==h:break
                if u.IsWindowVisible(window) and not u.IsIconic(window):
                    rect=W.RECT()
                    if not u.GetWindowRect(window,C.byref(rect)):return False
                    if rect.left<=x<rect.right and rect.top<=y<rect.bottom:return False
                window=u.GetWindow(window,2)
            else:return False
            hit=C.c_size_t()
            lparam=((y&0xffff)<<16)|(x&0xffff)
            success=u.SendMessageTimeoutW(h,0x0084,0,lparam,0x0002|0x0020,100,C.byref(hit))
            return bool(success) and hit.value==2
        point=None;tested=0;deadline=time.monotonic()+2
        for dy in (12,18,24,30,36,42,48,54):
            for fraction in (.5,.6,.4,.7,.3,.8,.2,.9):
                if time.monotonic()>deadline:break
                x=bounds.left+int((bounds.right-bounds.left)*fraction);y=bounds.top+dy
                tested+=1
                if point_is_caption(x,y):point=(x,y);break
            if point is not None or time.monotonic()>deadline:break
        if point is None:raise RuntimeError('Owned_browser_no_verified_caption_point')
        # Refuse held modifiers/buttons; never reset user state to manufacture
        # an uncontested click. This records no keystroke content.
        if any(u.GetAsyncKeyState(key)&0x8000 for key in (0x10,0x11,0x12,0x5b,0x5c,1,2,4)):
            raise RuntimeError('Owned_browser_caption_input_state_busy')
        class MOUSEINPUT(C.Structure):
            _fields_=[('dx',W.LONG),('dy',W.LONG),('mouseData',W.DWORD),('flags',W.DWORD),('time',W.DWORD),('extra',C.c_size_t)]
        class PAYLOAD(C.Union):_fields_=[('mouse',MOUSEINPUT)]
        class INPUT(C.Structure):_fields_=[('type',W.DWORD),('payload',PAYLOAD)]
        expected_size=40 if C.sizeof(C.c_void_p)==8 else 28
        if C.sizeof(INPUT)!=expected_size:raise RuntimeError('Owned_browser_caption_INPUT_layout_mismatch')
        u.SendInput.argtypes=[W.UINT,C.POINTER(INPUT),C.c_int];u.SendInput.restype=W.UINT
        def event(flags,dx=0,dy=0):
            item=INPUT();item.type=0;item.payload.mouse=MOUSEINPUT(dx,dy,0,flags,0,0);return item
        x,y=point
        if not point_is_caption(x,y):raise RuntimeError('Owned_browser_caption_point_changed_before_move')
        absolute=[round((x-vx)*65535/(vw-1)),round((y-vy)*65535/(vh-1))]
        record('move_prepared',intended_screen_point=[x,y],encoded_absolute=absolute,virtual_screen=[vx,vy,vw,vh])
        move=event(0x0001|0x8000|0x4000,*absolute);move_before=time.perf_counter_ns()
        inserted=u.SendInput(1,C.byref(move),C.sizeof(INPUT));move_after=time.perf_counter_ns()
        record('move_queued',intended_screen_point=[x,y],events_inserted=inserted,before_ns=move_before,after_ns=move_after)
        if inserted!=1:raise RuntimeError('Owned_browser_caption_pointer_move_refused')
        actual=W.POINT();settle_deadline=time.monotonic()+.5;polls=0
        # SendInput queues the MOVE. Observe actual cursor completion using the
        # unchanged own HWND/HTCAPTION/higher-Z predicate; never infer success.
        while True:
            observed=bool(u.GetCursorPos(C.byref(actual)));polls+=1
            verified=observed and point_is_caption(actual.x,actual.y)
            record('cursor_observed',intended_screen_point=[x,y],actual_screen_point=[actual.x,actual.y] if observed else None,get_cursor_succeeded=observed,own_HTCAPTION_verified=verified,poll=polls)
            if verified:break
            if time.monotonic()>=settle_deadline:raise RuntimeError('Owned_browser_actual_cursor_not_caption_within_500ms')
            time.sleep(.01)
        # Recheck immediately after recording, before one down/up pair. No
        # held user key/button is reset, and no other HWND receives a click.
        if not u.GetCursorPos(C.byref(actual)) or not point_is_caption(actual.x,actual.y):raise RuntimeError('Owned_browser_caption_changed_before_click')
        if any(u.GetAsyncKeyState(key)&0x8000 for key in (0x10,0x11,0x12,0x5b,0x5c,1,2,4)):raise RuntimeError('Owned_browser_caption_input_state_busy_before_click')
        pair=(INPUT*2)(event(0x0002),event(0x0004));before=time.perf_counter_ns()
        sent=u.SendInput(2,pair,C.sizeof(INPUT));after=time.perf_counter_ns()
        if sent==1:
            release=event(0x0004)
            if u.SendInput(1,C.byref(release),C.sizeof(INPUT))!=1:
                raise RuntimeError('Owned_browser_partial_caption_release_unproven')
        record("caption_click_queued",intended_screen_point=[x,y],actual_screen_point=[actual.x,actual.y],events_inserted=sent,before_ns=before,after_ns=after)
        if sent!=2:raise RuntimeError('Owned_browser_caption_click_refused_or_partial')
        return {'screen_point':[actual.x,actual.y],'verified_HTCAPTION':True,'owned_point_and_nonocclusion_rechecked':True,'candidate_count':tested,'normal_HWND_TOP_NOACTIVATE':True,'click_pairs':1,'events_inserted':sent,'before_input_ns':before,'after_input_ns':after,'foreground_gate_waived':False}

    def validate(self):
        u=self.user;h=self.hwnd
        if geometry(u,h)!=self.expected or self.expected['host_pid']!=self.pid:
            raise RuntimeError('Owned_browser_geometry_or_owner_changed')
        if not u.IsWindowVisible(h) or u.IsIconic(h) or u.GetAncestor(u.GetForegroundWindow(),2)!=h:
            raise RuntimeError('Owned_browser_not_foreground_and_visible')
        x,y=self.expected['origin'];width,height=self.expected['client']
        if width<=0 or height<=0:raise RuntimeError('Owned_browser_empty_client')
        # Monitor work-area containment rejects clipping by monitor edge/taskbar.
        class MONITORINFO(C.Structure):
            _fields_=[('size',W.DWORD),('monitor',W.RECT),('work',W.RECT),('flags',W.DWORD)]
        u.MonitorFromWindow.argtypes=[W.HWND,W.DWORD];u.MonitorFromWindow.restype=W.HANDLE
        u.GetMonitorInfoW.argtypes=[W.HANDLE,C.POINTER(MONITORINFO)];u.GetMonitorInfoW.restype=W.BOOL
        info=MONITORINFO();info.size=C.sizeof(info)
        if not u.GetMonitorInfoW(u.MonitorFromWindow(h,0),C.byref(info)) or not (info.work.left<=x and info.work.top<=y and x+width<=info.work.right and y+height<=info.work.bottom):
            raise RuntimeError('Owned_browser_client_outside_monitor_work_area')
        for fx,fy in [(0,0),(.5,0),(1,0),(0,.5),(.5,.5),(1,.5),(0,1),(.5,1),(1,1)]:
            point=W.POINT(x+int((width-1)*fx),y+int((height-1)*fy))
            if u.GetAncestor(u.WindowFromPoint(point),2)!=h:raise RuntimeError('Owned_browser_point_occluded')
        for name,restype,args in [('GetTopWindow',W.HWND,[W.HWND]),('GetWindow',W.HWND,[W.HWND,W.UINT]),('GetWindowRect',W.BOOL,[W.HWND,C.POINTER(W.RECT)])]:
            fn=getattr(u,name);fn.restype=restype;fn.argtypes=args
        other=u.GetTopWindow(None)
        for _ in range(1024):
            if not other:raise RuntimeError('Owned_browser_missing_from_Z_order')
            if int(other)==h:return
            if u.IsWindowVisible(other) and not u.IsIconic(other):
                rect=W.RECT()
                if not u.GetWindowRect(other,C.byref(rect)):raise RuntimeError('Occluding_window_geometry_unavailable')
                if rect.left<x+width and rect.right>x and rect.top<y+height and rect.bottom>y:
                    raise RuntimeError('Owned_browser_client_intersects_higher_Z_window')
            other=u.GetWindow(other,2)
        raise RuntimeError('Owned_browser_Z_order_bound_exceeded')

    async def check(self,label,capture=False):
        started=time.perf_counter_ns()
        state=await self.page.evaluate('({visible:document.visibilityState==="visible",focused:document.hasFocus(),page_ms:performance.now()})')
        if not state['visible'] or not state['focused']:raise RuntimeError('Owned_page_not_visible_and_focused')
        self.validate()
        receipt={'label':label,'python_before_ns':started,'page_state':state,**self.identity,'continuous_visibility_proven':False}
        if capture:
            width,height=self.expected['client']
            image=capture_owned_roi(self.user,self.kernel,self.gdi,self.hwnd,self.expected,(0,0,width,height),self.validate)
            pixels=image.pop('pixels')
            # stdlib bottom-up-independent BITMAPINFOHEADER: negative height is top-down.
            dib=struct.pack('<IiiHHIIiiII',40,width,-height,1,32,0,len(pixels),0,0,0,0)
            data=struct.pack('<2sIHHI',b'BM',14+40+len(pixels),0,0,54)+dib+pixels
            path=self.run/(label+'-owned-client.bmp');path.write_bytes(data)
            receipt['native_capture']={**image,'file':path.name,'file_sha256':hashlib.sha256(data).hexdigest()}
        self.validate();receipt['python_after_ns']=time.perf_counter_ns()
        return receipt
