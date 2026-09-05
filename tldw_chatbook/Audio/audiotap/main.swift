// tldw-audiotap: macOS 14.2+ system-audio capture helper.
// Emits 20 ms frames of PCM16 mono 16 kHz on stdout; READY on stderr once
// the IO proc runs; exits on stdin EOF / SIGTERM. Exit 2 = tap creation
// failed (usually the System Audio Recording permission), 3 = unsupported OS.
import AVFoundation
import CoreAudio
import Foundation

let frameBytes = 640
let ringSeconds = 2

final class Ring {
    private var buffer = [UInt8](repeating: 0, count: 32_000 * ringSeconds)
    private var head = 0, count = 0
    private let lock = NSLock()
    private(set) var dropped = 0

    func push(_ data: UnsafeRawBufferPointer) {
        lock.lock(); defer { lock.unlock() }
        for byte in data {
            if count == buffer.count { head = (head + 1) % buffer.count; count -= 1; dropped += 1 }
            buffer[(head + count) % buffer.count] = byte
            count += 1
        }
    }

    func pop(_ n: Int) -> [UInt8]? {
        lock.lock(); defer { lock.unlock() }
        guard count >= n else { return nil }
        var out = [UInt8](repeating: 0, count: n)
        for i in 0..<n { out[i] = buffer[(head + i) % buffer.count] }
        head = (head + n) % buffer.count
        count -= n
        return out
    }
}

func stderr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

guard #available(macOS 14.2, *) else { stderr("unsupported macOS"); exit(3) }

func processObject(for pid: pid_t) -> AudioObjectID? {
    var pidVar = pid
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = withUnsafePointer(to: &pidVar) { ptr in
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                   UInt32(MemoryLayout<pid_t>.size), ptr, &size, &object)
    }
    return status == noErr ? object : nil
}

var exclude: [AudioObjectID] = []
if let own = processObject(for: ProcessInfo.processInfo.processIdentifier) { exclude.append(own) }
if let parent = processObject(for: getppid()) { exclude.append(parent) }

let description = CATapDescription(stereoGlobalTapButExcludeProcesses: exclude)
description.uuid = UUID()
description.muteBehavior = .unmuted
description.name = "tldw-audiotap"
var tapID = AudioObjectID(kAudioObjectUnknown)
var status = AudioHardwareCreateProcessTap(description, &tapID)
guard status == noErr else { stderr("process tap failed: \(status) (grant System Audio Recording in Privacy & Security)"); exit(2) }

let aggregate: [String: Any] = [
    kAudioAggregateDeviceNameKey: "tldw-audiotap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapUIDKey: description.uuid.uuidString,
        kAudioSubTapDriftCompensationKey: true,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
status = AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID)
guard status == noErr else { stderr("aggregate device failed: \(status)"); exit(2) }

var formatAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
status = AudioObjectGetPropertyData(tapID, &formatAddr, 0, nil, &asbdSize, &asbd)
guard status == noErr, let inFormat = AVAudioFormat(streamDescription: &asbd) else { stderr("tap format failed: \(status)"); exit(2) }
guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true),
      let converter = AVAudioConverter(from: inFormat, to: outFormat) else { stderr("converter failed"); exit(2) }

let ring = Ring()
var procID: AudioDeviceIOProcID?
status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { _, inData, _, _, _ in
    let frames = AVAudioFrameCount(inData.pointee.mBuffers.mDataByteSize) / AVAudioFrameCount(max(1, asbd.mBytesPerFrame))
    guard frames > 0, let input = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: inData, deallocator: nil) else { return }
    input.frameLength = frames
    let capacity = AVAudioFrameCount(Double(frames) * 16_000.0 / inFormat.sampleRate) + 16
    guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, outStatus in
        if consumed { outStatus.pointee = .noDataNow; return nil }
        consumed = true; outStatus.pointee = .haveData; return input
    }
    guard error == nil, let bytes = output.int16ChannelData?.pointee else { return }
    let byteCount = Int(output.frameLength) * 2
    ring.push(UnsafeRawBufferPointer(start: bytes, count: byteCount))
}
guard status == noErr, let ioProc = procID else { stderr("io proc failed: \(status)"); exit(2) }
status = AudioDeviceStart(aggregateID, ioProc)
guard status == noErr else { stderr("device start failed: \(status)"); exit(2) }
stderr("READY")

let writer = Thread {
    let out = FileHandle.standardOutput
    var reported = 0
    while true {
        if let frame = ring.pop(frameBytes) {
            out.write(Data(frame))
        } else {
            usleep(5_000)
        }
        if ring.dropped - reported >= 32_000 { reported = ring.dropped; stderr("dropped \(reported) bytes") }
    }
}
writer.start()

signal(SIGTERM) { _ in exit(0) }
signal(SIGPIPE) { _ in exit(0) }
// Block until the parent closes our stdin.
_ = FileHandle.standardInput.readDataToEndOfFile()
AudioDeviceStop(aggregateID, ioProc)
AudioDeviceDestroyIOProcID(aggregateID, ioProc)
AudioHardwareDestroyAggregateDevice(aggregateID)
AudioHardwareDestroyProcessTap(tapID)
exit(0)
