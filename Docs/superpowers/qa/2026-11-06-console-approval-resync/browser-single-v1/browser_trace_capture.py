"""Bounded page-scoped Chromium trace plus explicit page/trace/server brackets."""
import asyncio
import json
import secrets
from browser_presented_frames import offset_interval

def _samples(value):
    if type(value) is not int or value not in (1,20):
        raise ValueError('Exact_calibration_samples_1_or20_required')


class BrowserTraceCapture:
    def __init__(self, page, cdp):
        self.page = page
        self.cdp = cdp
        self.events = []
        self.mark_brackets = []
        self.http_brackets = []
        self.complete = None
        self.cdp.on('Tracing.dataCollected',self._data)
        self.cdp.on('Tracing.tracingComplete',self._complete)

    def _data(self, value):
        self.events.extend(value['value'])

    def _complete(self, value):
        if self.complete is not None and not self.complete.done():
            self.complete.set_result(value)

    async def start(self, *, calibration_samples=20):
        _samples(calibration_samples)
        self.events = []
        self.mark_brackets = []
        self.http_brackets = []
        self.complete = asyncio.get_running_loop().create_future()
        await self.cdp.send('Tracing.start',{'transferMode':'ReportEvents','screenshotMaxSize':1500,'screenshotMaxCount':50,'traceConfig':{
            'recordMode':'recordContinuously','traceBufferSizeInKb':8192,
            # The benchmark group member enables PipelineReporter, FramePresented
            # and Display::FrameDisplayed without the broad compositor categories.
            'includedCategories':['benchmark','blink.user_timing','disabled-by-default-devtools.screenshot']}})
        await self.calibrate('before', samples=calibration_samples)

    async def calibrate(self, phase, *, samples=20):
        _samples(samples)
        for _ in range(samples):
            name = 'qa-clock-' + secrets.token_hex(12)
            bracket = await self.page.evaluate("""name => { const before_ms=performance.now(); performance.mark(name); const after_ms=performance.now(); return {before_ms,after_ms}; }""",name)
            self.mark_brackets.append({'name':name,'phase':phase,**bracket})
            remote = await self.page.evaluate("""async () => { const before_ms=performance.now(); const response=await fetch('/qa/clock',{cache:'no-store'}); const data=await response.json(); const after_ms=performance.now(); return {before_ms,after_ms,server_ns:data.server_ns}; }""")
            self.http_brackets.append({'phase':phase,**remote})

    async def stop(self, output, *, calibration_samples=20):
        _samples(calibration_samples)
        await self.calibrate('after', samples=calibration_samples)
        await self.cdp.send('Tracing.end')
        status = await asyncio.wait_for(self.complete,30)
        # Retain raw evidence before refusing a lost/inconsistent boundary.
        output.write_text(json.dumps({'traceEvents':self.events,'tracing_complete':status,'page_mark_brackets':self.mark_brackets,'server_http_brackets':self.http_brackets}),encoding='utf-8')
        assert not status.get('dataLossOccurred'), 'trace_data_loss'
        marks = {}
        for event in self.events:
            marks.setdefault(event.get('name'),[]).append(event)
        mark_samples = []
        pids = set()
        for bracket in self.mark_brackets:
            matches = marks.get(bracket['name'],[])
            assert len(matches)==1,'missing_or_ambiguous_trace_clock_mark'
            event = matches[0]
            pids.add(event['pid'])
            mark_samples.append({**bracket,'remote_ms':event['ts']/1000})
        assert len(pids)==1,'clock_marks_changed_renderer'
        intervals = {}
        for phase in ('before','after'):
            intervals[phase] = {
                'trace_page':offset_interval([sample for sample in mark_samples if sample['phase']==phase]),
                'server_page':offset_interval([{**sample,'remote_ms':sample['server_ns']/1e6} for sample in self.http_brackets if sample['phase']==phase])}
        envelope = {}
        for clock in ('trace_page','server_page'):
            values = [intervals[phase][clock] for phase in ('before','after')]
            envelope[clock] = {'lower_ms':min(value['lower_ms'] for value in values),'upper_ms':max(value['upper_ms'] for value in values)}
            envelope[clock]['width_ms'] = envelope[clock]['upper_ms']-envelope[clock]['lower_ms']
            before, after = values
            envelope[clock]['observed_offset_shift_interval_ms'] = [after['lower_ms']-before['upper_ms'], after['upper_ms']-before['lower_ms']]
            envelope[clock]['observed_offset_center_shift_ms'] = ((after['lower_ms']+after['upper_ms'])-(before['lower_ms']+before['upper_ms']))/2
        return {'page_pid':pids.pop(),'intervals':intervals,'envelope':envelope,'trace_event_count':len(self.events),'known_clock_error_upper_ms':envelope['trace_page']['width_ms']+envelope['server_page']['width_ms']}
