/// <reference lib="dom" />

import type {
  Event,
  RealtimeClientEvents,
  RealtimeCustomEvents,
  RealtimeServerEvents
} from './events'
import type {
  EventHandlerResult,
  FormattedTool,
  Realtime,
  ToolHandler
} from './types'
import { RealtimeConversation } from './conversation'
import { RealtimeEventHandler } from './event-handler'
import { arrayBufferToBase64, assert, mergeInt16Arrays } from './utils'

/**
 * The WebRTCClient class is the main interface for interacting with the
 * OpenAI Realtime API via WebRTC. It handles connection, configuration, conversation
 * updates, and server event handling.
 */
export class WebRTCClient extends RealtimeEventHandler<
  | RealtimeClientEvents.EventType
  | RealtimeServerEvents.EventType
  | RealtimeCustomEvents.EventType,
  Event,
  RealtimeClientEvents.EventMap &
    RealtimeServerEvents.EventMap &
    RealtimeCustomEvents.EventMap
> {
  readonly defaultSessionConfig: Realtime.SessionConfig
  sessionConfig: Realtime.SessionConfig

  readonly relay: boolean

  conversation: RealtimeConversation
  peerConnection: RTCPeerConnection | null = null
  dataChannel: RTCDataChannel | null = null
  mediaStream: MediaStream | null = null
  audioElement: HTMLAudioElement | null = null

  inputAudioBuffer: Int16Array
  sessionCreated: boolean
  tools: Record<
    string,
    {
      definition: Realtime.ToolDefinition
      handler: ToolHandler
    }
  >

  /** The ephemeral token to use for WebRTC connection */
  ephemeralToken?: string

  /** The model to use for the WebRTC connection */
  private readonly model: string

  /** Audio context and analyzer for level monitoring */
  private audioContext: AudioContext | null = null
  private analyzer: AnalyserNode | null = null

  constructor({
    sessionConfig,
    relay = false,
    apiKey: _apiKey,
    model = 'gpt-4-realtime-preview-2024-12-17',
    url: _url = 'https://api.openai.com/v1/realtime',
    dangerouslyAllowAPIKeyInBrowser: _dangerouslyAllowAPIKeyInBrowser = false,
    debug = false,
    ephemeralToken
  }: {
    sessionConfig?: Partial<Omit<Realtime.SessionConfig, 'tools'>>
    apiKey?: string
    model?: string
    url?: string
    dangerouslyAllowAPIKeyInBrowser?: boolean
    debug?: boolean
    relay?: boolean
    /** The ephemeral token to use for WebRTC connection */
    ephemeralToken?: string
  } = {}) {
    super()

    this.defaultSessionConfig = {
      modalities: ['text', 'audio'],
      voice: 'alloy',
      input_audio_format: 'pcm16',
      output_audio_format: 'pcm16',
      input_audio_transcription: {
        model: 'whisper-1'
      },
      turn_detection: null,
      tools: [],
      tool_choice: 'auto',
      temperature: 0.8,
      max_response_output_tokens: 4096,
      ...sessionConfig
    }

    this.sessionConfig = {}
    this.sessionCreated = false
    this.tools = {}
    this.inputAudioBuffer = new Int16Array(0)
    this.relay = !!relay
    this.ephemeralToken = ephemeralToken
    this.model = model
    this.audioElement = document.createElement('audio')
    this.audioElement.autoplay = true

    this.conversation = new RealtimeConversation({ debug })

    this._resetConfig()
  }

  /**
   * Resets sessionConfig and conversation to defaults.
   */
  protected _resetConfig() {
    this.sessionCreated = false
    this.tools = {}
    this.sessionConfig = structuredClone(this.defaultSessionConfig)
    this.inputAudioBuffer = new Int16Array(0)
  }

  /**
   * Whether the WebRTC connection is established and session is created.
   */
  get isConnected(): boolean {
    return this.dataChannel?.readyState === 'open' && this.sessionCreated
  }

  /**
   * Whether the client is in relay mode. When in relay mode, the client will
   * not attempt to invoke tools.
   */
  get isRelay(): boolean {
    return this.relay
  }

  /**
   * Resets the client instance entirely: disconnects and clears configs.
   */
  reset() {
    this.disconnect()
    this.clearEventHandlers()
    this._resetConfig()
  }

  /**
   * Connects to the OpenAI Realtime API via WebRTC and updates the session config.
   */
  async connect() {
    if (this.isConnected) {
      return
    }

    assert(
      this.ephemeralToken,
      'Ephemeral token is required for WebRTC connection'
    )

    // Create a peer connection
    this.peerConnection = new RTCPeerConnection()

    // Set up connection promise to wait for both WebRTC connection and session creation
    const connectionPromise = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'))
      }, 10_000) // 10 second timeout

      if (!this.peerConnection) return reject(new Error('No peer connection'))

      const handler = (event: any) => {
        if (event.event?.type === 'session.created') {
          this.off('realtime.event', handler)
          this.sessionCreated = true
          clearTimeout(timeout)
          resolve()
        }
      }
      this.on('realtime.event', handler)

      this.peerConnection.onconnectionstatechange = () => {
        if (this.peerConnection?.connectionState === 'failed') {
          clearTimeout(timeout)
          reject(new Error('Connection failed'))
        }
      }
    })

    // Set up to receive remote audio from the model
    this.peerConnection.ontrack = (e: RTCTrackEvent) => {
      if (e.track.kind === 'audio' && e.streams[0] && this.audioElement) {
        const stream = e.streams[0]
        this.audioElement.srcObject = stream

        // Set up audio analysis
        this.audioContext = new AudioContext()
        const source = this.audioContext.createMediaStreamSource(stream)
        this.analyzer = this.audioContext.createAnalyser()
        this.analyzer.fftSize = 2048
        source.connect(this.analyzer)
        source.connect(this.audioContext.destination) // For audio playback

        console.log('Audio context created')
        console.log(this.analyzer)
      }
    }

    // Add local audio track for microphone input
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true
      })
      const track = this.mediaStream.getTracks()[0]
      if (track) {
        this.peerConnection.addTrack(track, this.mediaStream)
      }
    } catch (err) {
      console.error('Failed to get user media:', err)
      throw err
    }

    // Set up data channel for sending and receiving events
    this.dataChannel = this.peerConnection.createDataChannel('oai-events')
    this._setupDataChannelHandlers()

    // Start the session using SDP
    const offer = await this.peerConnection.createOffer()
    await this.peerConnection.setLocalDescription(offer)

    const baseUrl = 'https://api.openai.com/v1/realtime'
    const sdpResponse = await fetch(`${baseUrl}?model=${this.model}`, {
      method: 'POST',
      body: offer.sdp,
      headers: {
        Authorization: `Bearer ${this.ephemeralToken}`,
        'Content-Type': 'application/sdp'
      }
    })

    const answer: RTCSessionDescriptionInit = {
      type: 'answer',
      sdp: await sdpResponse.text()
    }
    await this.peerConnection.setRemoteDescription(answer)

    // Wait for connection to be established
    await connectionPromise

    this.updateSession()
  }

  /**
   * Sets up event handlers for the WebRTC data channel
   */
  protected _setupDataChannelHandlers() {
    if (!this.dataChannel) return

    const handler = (
      event: RealtimeServerEvents.ServerEvent,
      ...args: any[]
    ): EventHandlerResult => {
      if (!this.isConnected) return {}
      return this.conversation.processEvent(event, ...args)
    }

    const handlerWithDispatch = (
      event: RealtimeServerEvents.ServerEvent,
      ...args: any[]
    ) => {
      const res = handler(event, ...args)

      if (res.item) {
        // FIXME: This is only here because `item.input_audio_transcription.completed`
        // can fire before `item.created`, resulting in empty item. This happens in
        // VAD mode with empty audio.
        this.dispatch('conversation.updated', {
          type: 'conversation.updated',
          ...res
        })
      }

      return res
    }

    this.dataChannel.addEventListener('message', (e: MessageEvent) => {
      const event = JSON.parse(e.data) as RealtimeServerEvents.ServerEvent
      this.dispatch('realtime.event', {
        type: 'realtime.event',
        time: new Date().toISOString(),
        source: 'server',
        event
      })

      // Handle all server events
      switch (event.type) {
        case 'session.created':
          this.sessionCreated = true
          break

        case 'session.updated':
          // Update session config
          this.sessionConfig = {
            ...this.sessionConfig,
            ...(event as RealtimeServerEvents.SessionUpdatedEvent).session
          }
          break

        case 'conversation.item.created':
          const res = handlerWithDispatch(event)
          if (!res.item) return

          this.dispatch('conversation.item.appended', {
            type: 'conversation.item.appended',
            ...res
          })

          if (res.item.status === 'completed') {
            this.dispatch('conversation.item.completed', {
              type: 'conversation.item.completed',
              ...res
            })
          }
          break

        case 'conversation.item.truncated':
          handlerWithDispatch(event)
          break

        case 'conversation.item.deleted':
          handlerWithDispatch(event)
          break

        case 'input_audio_buffer.committed':
          // Skip processing this event as it's not handled by conversation
          break

        case 'input_audio_buffer.speech_started':
          handler(event)
          this.dispatch('conversation.interrupted', event)
          break

        case 'input_audio_buffer.speech_stopped':
          handler(event, this.inputAudioBuffer)
          break

        case 'response.created':
          handler(event)
          break

        case 'response.done':
          // Skip processing this event as it's not handled by conversation
          break

        case 'response.output_item.added':
          handler(event)
          break

        case 'response.output_item.done':
          const outputRes = handlerWithDispatch(event)
          if (!outputRes.item?.formatted) return

          if (outputRes.item.status === 'completed') {
            this.dispatch('conversation.item.completed', {
              type: 'conversation.item.completed',
              ...outputRes
            })
          }

          if (outputRes.item.formatted.tool) {
            this._handleToolCall(outputRes.item.formatted.tool)
          }
          break

        case 'response.content_part.added':
          handler(event)
          break

        case 'response.content_part.done':
          // Skip processing this event as it's not handled by conversation
          break

        case 'response.text.delta':
          handler(event)
          break

        case 'response.text.done':
          handler(event)
          break

        case 'response.audio_transcript.delta':
          handler(event)
          break

        case 'response.audio_transcript.done':
          // Skip processing this event as it's not handled by conversation
          break

        case 'response.audio.delta':
          const audioRes = handler(event)
          this.dispatch('conversation.updated', {
            type: 'conversation.updated',
            delta: event,
            ...audioRes
          })
          break

        case 'response.audio.done':
          // Skip processing this event as it's not handled by conversation
          break

        case 'response.function_call_arguments.delta':
          handler(event)
          break

        case 'response.function_call_arguments.done':
          handler(event)
          break

        case 'rate_limits.updated':
          // No need to process rate limits in conversation
          break

        case 'error':
          console.error(
            'Server error:',
            (event as RealtimeServerEvents.ErrorEvent).error
          )
          break

        default:
          console.warn('Unhandled server event:', event.type)
      }
    })

    this.dataChannel.addEventListener('open', () => {
      this.dispatch('realtime.event', {
        type: 'realtime.event',
        time: new Date().toISOString(),
        source: 'client',
        event: { type: 'client.connected' }
      })
    })

    this.dataChannel.addEventListener('close', () => {
      this.dispatch('realtime.event', {
        type: 'realtime.event',
        time: new Date().toISOString(),
        source: 'client',
        event: { type: 'client.disconnected' }
      })
    })
  }

  /**
   * Handles tool calls by executing the tool and sending the result
   */
  protected async _handleToolCall(tool: FormattedTool) {
    try {
      const jsonArguments = JSON.parse(tool.arguments)
      const toolConfig = this.tools[tool.name]
      if (!toolConfig) {
        console.warn(`Tool "${tool.name}" not found`)
        return
      }

      const result = await Promise.resolve(toolConfig.handler(jsonArguments))
      this._sendEvent('conversation.item.create', {
        item: {
          type: 'function_call_output',
          call_id: tool.call_id,
          output: JSON.stringify(result)
        }
      })
    } catch (err: any) {
      console.warn(`Error calling tool "${tool.name}":`, err.message)

      this._sendEvent('conversation.item.create', {
        item: {
          type: 'function_call_output',
          call_id: tool.call_id,
          output: JSON.stringify({ error: err.message })
        }
      })
    }

    this.createResponse()
  }

  /**
   * Sends an event through the WebRTC data channel
   */
  protected _sendEvent(type: string, payload: any = {}) {
    if (!this.isConnected) return

    const event = {
      type,
      event_id: crypto.randomUUID(),
      ...payload
    }

    this.dataChannel?.send(JSON.stringify(event))
    this.dispatch('realtime.event', {
      type: 'realtime.event',
      time: new Date().toISOString(),
      source: 'client',
      event
    })
  }

  /**
   * Disconnects from the WebRTC connection and clears the conversation history.
   */
  disconnect() {
    if (this.dataChannel) {
      this.dataChannel.close()
      this.dataChannel = null
    }

    if (this.peerConnection) {
      this.peerConnection.close()
      this.peerConnection = null
    }

    if (this.mediaStream) {
      for (const track of this.mediaStream.getTracks()) {
        track.stop()
      }
      this.mediaStream = null
    }

    this.sessionCreated = false
    this.conversation.clear()
  }

  /**
   * Gets the active turn detection mode.
   */
  getTurnDetectionType(): 'server_vad' | undefined {
    return this.sessionConfig.turn_detection?.type
  }

  /**
   * Adds a tool to the session.
   */
  addTool(definition: Realtime.PartialToolDefinition, handler: ToolHandler) {
    assert(!this.isRelay, 'Unable to add tools in relay mode')
    assert(definition?.name, 'Missing tool name in definition')
    const { name } = definition

    assert(
      typeof handler === 'function',
      `Tool "${name}" handler must be a function`
    )

    this.tools[name] = {
      definition: {
        type: 'function',
        ...definition
      },
      handler
    }
    this.updateSession()
  }

  /**
   * Removes a tool from the session.
   */
  removeTool(name: string) {
    assert(!this.isRelay, 'Unable to add tools in relay mode')
    assert(
      this.tools[name],
      `Tool "${name}" does not exist, can not be removed.`
    )
    delete this.tools[name]
    this.updateSession()
  }

  /**
   * Deletes an item.
   */
  deleteItem(id: string) {
    this._sendEvent('conversation.item.delete', { item_id: id })
  }

  /**
   * Updates session configuration.
   *
   * If the client is not yet connected, the session will be updated upon connection.
   */
  updateSession(sessionConfig: Realtime.SessionConfig = {}) {
    const tools = Object.values(this.tools).map(({ definition }) => definition)

    this.sessionConfig = {
      ...this.sessionConfig,
      ...sessionConfig,
      tools
    }

    if (this.isConnected && !this.isRelay) {
      this._sendEvent('session.update', {
        session: structuredClone(this.sessionConfig)
      })
    }
  }

  /**
   * Sends user message content and generates a response.
   */
  sendUserMessageContent(
    content: Array<
      Realtime.InputTextContentPart | Realtime.InputAudioContentPart
    >
  ) {
    assert(!this.isRelay, 'Unable to send messages directly in relay mode')

    if (content.length) {
      this._sendEvent('conversation.item.create', {
        item: {
          type: 'message',
          role: 'user',
          content
        }
      })
    }

    this.createResponse()
  }

  /**
   * Appends user audio to the existing audio buffer.
   */
  appendInputAudio(arrayBuffer: Int16Array | ArrayBuffer) {
    assert(!this.isRelay, 'Unable to append input audio directly in relay mode')

    if (arrayBuffer.byteLength > 0) {
      this._sendEvent('input_audio_buffer.append', {
        audio: arrayBufferToBase64(arrayBuffer)
      })

      this.inputAudioBuffer = mergeInt16Arrays(
        this.inputAudioBuffer,
        arrayBuffer
      )
    }
  }

  /**
   * Forces the model to generate a response.
   */
  createResponse() {
    assert(!this.isRelay, 'Unable to create a response directly in relay mode')

    if (!this.getTurnDetectionType() && this.inputAudioBuffer.byteLength > 0) {
      this._sendEvent('input_audio_buffer.commit')
      this.conversation.queueInputAudio(this.inputAudioBuffer)
      this.inputAudioBuffer = new Int16Array(0)
    }

    this._sendEvent('response.create')
  }

  /**
   * Cancels the ongoing server generation and truncates ongoing generation, if
   * applicable.
   *
   * If no id provided, will simply call `cancel_generation` command.
   */
  cancelResponse(
    /** The ID of the item to cancel. */
    id?: string,
    /** The number of samples to truncate past for the ongoing generation. */
    sampleCount = 0
  ): Realtime.AssistantItem | undefined {
    assert(!this.isRelay, 'Unable to cancel a response directly in relay mode')

    if (!id) {
      this._sendEvent('response.cancel')
      return
    }

    const item = this.conversation.getItem(id)
    assert(item, `Could not find item "${id}"`)
    assert(
      item.type === 'message',
      `Can only cancelResponse messages with type "message"`
    )
    assert(
      item.role === 'assistant',
      `Can only cancelResponse messages with role "assistant"`
    )

    this._sendEvent('response.cancel')
    const audioIndex = item.content.findIndex((c) => c.type === 'audio')
    assert(audioIndex >= 0, `Could not find audio on item ${id} to cancel`)

    this._sendEvent('conversation.item.truncate', {
      item_id: id,
      content_index: audioIndex,
      audio_end_ms: Math.floor(
        (sampleCount / this.conversation.defaultFrequency) * 1000
      )
    })

    return item
  }

  /**
   * Utility for waiting for the next `conversation.item.appended` event to be
   * triggered by the server.
   */
  async waitForNextItem(): Promise<Realtime.Item> {
    const event = await this.waitForNext('conversation.item.appended')
    return event.item
  }

  /**
   * Utility for waiting for the next `conversation.item.completed` event to be
   * triggered by the server.
   */
  async waitForNextCompletedItem(): Promise<Realtime.Item> {
    const event = await this.waitForNext('conversation.item.completed')
    return event.item
  }

  /**
   * Gets the current audio levels split into 8 frequency bins
   * @returns {number[]} Array of 8 numbers between 0 and 1 representing audio levels
   */
  getLevels(): number[] {
    if (!this.analyzer) {
      console.log('No analyzer found')
      return Array(8).fill(0)
    }

    const frequencyData = new Uint8Array(this.analyzer.frequencyBinCount)
    this.analyzer.getByteFrequencyData(frequencyData)

    // Split frequency data into 8 bins
    const levels: number[] = []
    const binSize = Math.floor(frequencyData.length / 8)

    for (let i = 0; i < 8; i++) {
      const start = i * binSize
      const end = start + binSize
      let sum = 0

      for (let j = start; j < end; j++) {
        const value = frequencyData[j] || 0
        sum += value
      }

      // Normalize to 0-1
      levels[i] = sum / (binSize * 255)
    }

    return levels
  }
}
