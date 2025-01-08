import 'dotenv/config'

import { WebRTCClient } from './webrtc-client'
import type { Event } from './events'
import type { Realtime } from './types'
import { trimDebugEvent } from './utils'
import { beforeEach, describe, expect, test, vi } from 'vitest'

// Mock browser APIs
const mockMediaStream = {
  getTracks: () => [
    {
      stop: vi.fn()
    }
  ]
}

const mockPeerConnection = {
  createOffer: vi.fn().mockResolvedValue({ sdp: 'mock-sdp' }),
  setLocalDescription: vi.fn(),
  setRemoteDescription: vi.fn(),
  close: vi.fn(),
  addTrack: vi.fn(),
  ontrack: null as any,
  onconnectionstatechange: null as any,
  connectionState: 'new',
  generateCertificate: vi.fn()
}

// Mock global browser APIs
vi.stubGlobal(
  'RTCPeerConnection',
  vi.fn().mockImplementation(() => mockPeerConnection)
)
vi.stubGlobal(
  'MediaStream',
  vi.fn().mockImplementation(() => mockMediaStream)
)
vi.stubGlobal('navigator', {
  mediaDevices: {
    getUserMedia: vi.fn().mockResolvedValue(mockMediaStream)
  }
})

vi.stubGlobal('crypto', {
  randomUUID: () => '123e4567-e89b-12d3-a456-426614174000'
})

// Mock fetch for SDP exchange
vi.stubGlobal(
  'fetch',
  vi.fn().mockImplementation(() =>
    Promise.resolve({
      text: () => Promise.resolve('mock-answer-sdp')
    })
  )
)

// Mock AudioContext and AnalyserNode
const mockAnalyser = {
  fftSize: 2048,
  frequencyBinCount: 1024,
  getByteFrequencyData: vi.fn(),
  connect: vi.fn()
}

const mockAudioContext = {
  createAnalyser: vi.fn().mockReturnValue(mockAnalyser),
  createMediaStreamSource: vi.fn().mockReturnValue({
    connect: vi.fn()
  }),
  destination: {}
}

vi.stubGlobal(
  'AudioContext',
  vi.fn().mockImplementation(() => mockAudioContext)
)
vi.stubGlobal('AnalyserNode', vi.fn())

describe('WebRTCClient', () => {
  let client: WebRTCClient
  const events: Event[] = []

  beforeEach(() => {
    vi.clearAllMocks()
    events.length = 0

    client = new WebRTCClient({
      debug: true,
      ephemeralToken: 'mock-token',
      sessionConfig: {
        instructions: 'Test instructions',
        turn_detection: null
      }
    })

    client.on('realtime.event', (event) => {
      events.push(trimDebugEvent(event.event))
    })
  })

  test('initialization', () => {
    expect(client.isConnected).toBe(false)
    expect(client.isRelay).toBe(false)
    expect(client.sessionConfig.instructions).toBe('Test instructions')
  })

  test('connect establishes WebRTC connection', async () => {
    await client.connect()

    expect(vi.mocked(RTCPeerConnection)).toHaveBeenCalled()
    expect(mockPeerConnection.createOffer).toHaveBeenCalled()
    expect(mockPeerConnection.setLocalDescription).toHaveBeenCalled()
    expect(mockPeerConnection.setRemoteDescription).toHaveBeenCalled()
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({
      audio: true
    })
  })

  test('disconnect cleans up resources', async () => {
    await client.connect()
    client.disconnect()

    expect(mockPeerConnection.close).toHaveBeenCalled()
    const tracks = mockMediaStream.getTracks()
    const track = tracks[0]
    if (track) {
      expect(track.stop).toHaveBeenCalled()
    }
    expect(client.isConnected).toBe(false)
  })

  test('audio levels are calculated correctly', async () => {
    await client.connect()

    // Mock frequency data
    const mockFrequencyData = new Uint8Array(1024).fill(128) // Mid-level frequencies
    mockAnalyser.getByteFrequencyData.mockImplementation((array) => {
      array.set(mockFrequencyData)
    })

    const levels = client.getLevels()
    expect(levels).toHaveLength(8)
    expect(levels.every((level) => level >= 0 && level <= 1)).toBe(true)
  })

  test('tool handling', async () => {
    const mockTool = {
      name: 'test-tool',
      description: 'A test tool',
      parameters: {
        type: 'object',
        properties: {
          test: { type: 'string' }
        }
      }
    }

    const mockHandler = vi.fn().mockResolvedValue({ result: 'success' })

    client.addTool(mockTool, mockHandler)

    const tool = client.tools['test-tool']
    expect(tool).toBeDefined()
    if (tool) {
      expect(tool.definition.name).toBe('test-tool')
    }

    client.removeTool('test-tool')
    expect(client.tools['test-tool']).toBeUndefined()
  })

  test('sending user message content', async () => {
    await client.connect()

    const content: Realtime.InputAudioContentPart[] = [
      {
        type: 'input_audio',
        audio: 'base64-audio-data'
      }
    ]

    client.sendUserMessageContent(content)

    // Verify the event was dispatched with correct type
    const createEvent = events.find(
      (e) => e.type === 'conversation.item.create'
    ) as any
    expect(createEvent).toBeDefined()
    expect(createEvent?.item?.content?.[0]?.type).toBe('input_audio')
  })
})
