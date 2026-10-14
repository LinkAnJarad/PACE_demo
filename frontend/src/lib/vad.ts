/**
 * Voice Activity Detection (VAD) Controller for Continuous Hands-Free Recording.
 *
 * Implements:
 * - Real-time Web Audio API energy & RMS calculation.
 * - 80Hz high-pass rumble filter to suppress desk/mic vibrations.
 * - Dynamic adaptive noise floor estimation.
 * - Multi-tier sensitivity presets (1: low/noisy -> 5: high/quiet).
 * - Silence hangover window detection before utterance commit.
 * - Zero-gain loopback to prevent acoustic feedback / echo through speakers.
 */

export type VadState = 'IDLE' | 'LISTENING' | 'SPEECH_DETECTED' | 'PROCESSING' | 'ROBOT_SPEAKING';

export interface VadPreset {
  mult: number;
  minFloor: number;
}

export const VAD_SENSITIVITY_PRESETS: Record<number, VadPreset> = {
  1: { mult: 2.2, minFloor: 0.009 },  // Low sensitivity (noisy background)
  2: { mult: 1.8, minFloor: 0.006 },
  3: { mult: 1.45, minFloor: 0.004 }, // Medium sensitivity (balanced default)
  4: { mult: 1.25, minFloor: 0.0025 },
  5: { mult: 1.1, minFloor: 0.0018 }, // High sensitivity (quiet room)
};

export interface VadCallbacks {
  onSpeechStart?: () => void;
  onSpeechEnd?: (speechDurationMs: number) => void;
  onRmsUpdate?: (rms: number, threshold: number, noiseFloor: number) => void;
  onStateChange?: (state: VadState) => void;
}

export interface VadOptions extends VadCallbacks {
  sensitivity?: number; // 1 to 5
  silenceHangoverMs?: number; // ms of silence before committing turn
  minSpeechDurationMs?: number; // min speech duration in ms (default: 300)
}

export class VadController {
  private stream: MediaStream;
  private audioContext: AudioContext | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private analyserNode: AnalyserNode | null = null;
  private muteGain: GainNode | null = null;

  private state: VadState = 'IDLE';
  private sensitivity: number = 3;
  private silenceHangoverMs: number = 850;
  private minSpeechDurationMs: number = 300;

  private isSpeaking: boolean = false;
  private speechStartTime: number = 0;
  private silenceStartTime: number = 0;
  private micStartupTime: number = 0;
  private noiseFloor: number = 0.003;
  private isRobotSpeaking: boolean = false;
  private isProcessing: boolean = false;

  private callbacks: VadCallbacks;

  constructor(stream: MediaStream, options?: VadOptions) {
    this.stream = stream;
    this.callbacks = {
      onSpeechStart: options?.onSpeechStart,
      onSpeechEnd: options?.onSpeechEnd,
      onRmsUpdate: options?.onRmsUpdate,
      onStateChange: options?.onStateChange,
    };

    if (options?.sensitivity) this.sensitivity = options.sensitivity;
    if (options?.silenceHangoverMs) this.silenceHangoverMs = options.silenceHangoverMs;
    if (options?.minSpeechDurationMs) this.minSpeechDurationMs = options.minSpeechDurationMs;
  }

  public getState(): VadState {
    return this.state;
  }

  private setState(newState: VadState) {
    if (this.state !== newState) {
      this.state = newState;
      this.callbacks.onStateChange?.(newState);
    }
  }

  public setSensitivity(level: number) {
    this.sensitivity = Math.max(1, Math.min(5, level));
  }

  public setSilenceHangoverMs(ms: number) {
    this.silenceHangoverMs = Math.max(300, Math.min(3000, ms));
  }

  public setRobotSpeaking(speaking: boolean) {
    this.isRobotSpeaking = speaking;
    if (speaking) {
      this.isSpeaking = false;
      this.silenceStartTime = 0;
      this.setState('ROBOT_SPEAKING');
    } else {
      if (!this.isProcessing) {
        this.setState('LISTENING');
      }
    }
  }

  public setProcessing(processing: boolean) {
    this.isProcessing = processing;
    if (processing) {
      this.isSpeaking = false;
      this.silenceStartTime = 0;
      this.setState('PROCESSING');
    } else {
      if (!this.isRobotSpeaking) {
        this.setState('LISTENING');
      }
    }
  }

  public start() {
    if (this.audioContext) return;

    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.audioContext = new AudioCtx();

      const source = this.audioContext.createMediaStreamSource(this.stream);

      // 80Hz rumble filter
      const highpass = this.audioContext.createBiquadFilter();
      highpass.type = 'highpass';
      highpass.frequency.value = 80;

      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 512;

      this.scriptProcessor = this.audioContext.createScriptProcessor(2048, 1, 1);
      this.scriptProcessor.onaudioprocess = this.handleAudioProcess.bind(this);

      source.connect(highpass);
      highpass.connect(this.analyserNode);
      this.analyserNode.connect(this.scriptProcessor);

      // Zero-gain loopback to prevent speaker feedback buzzing
      this.muteGain = this.audioContext.createGain();
      this.muteGain.gain.value = 0.0;
      this.scriptProcessor.connect(this.muteGain);
      this.muteGain.connect(this.audioContext.destination);

      this.micStartupTime = performance.now();
      this.isSpeaking = false;
      this.silenceStartTime = 0;
      this.setState('LISTENING');
    } catch (err) {
      console.error('[VAD] Failed to initialize AudioContext:', err);
      this.setState('IDLE');
    }
  }

  public stop() {
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor = null;
    }
    if (this.muteGain) {
      this.muteGain.disconnect();
      this.muteGain = null;
    }
    if (this.analyserNode) {
      this.analyserNode.disconnect();
      this.analyserNode = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }

    this.isSpeaking = false;
    this.silenceStartTime = 0;
    this.setState('IDLE');
  }

  private handleAudioProcess(e: AudioProcessingEvent) {
    if (this.state === 'IDLE') return;

    const now = performance.now();

    // 1. Warmup period: wait 600ms for hardware / AGC to settle while estimating initial noise floor
    if (now - this.micStartupTime < 600) {
      const inputData = e.inputBuffer.getChannelData(0);
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
      const rms = Math.sqrt(sum / inputData.length);
      this.noiseFloor = 0.90 * this.noiseFloor + 0.10 * rms;
      return;
    }

    // 2. Echo suppression: ignore microphone input while robot speech audio is playing or backend processing
    if (this.isRobotSpeaking || this.isProcessing) {
      return;
    }

    const inputData = e.inputBuffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < inputData.length; i++) {
      sum += inputData[i] * inputData[i];
    }
    const rms = Math.sqrt(sum / inputData.length);

    // Adaptive noise floor tracking when NOT speaking
    if (!this.isSpeaking) {
      this.noiseFloor = 0.98 * this.noiseFloor + 0.02 * rms;
    }

    const preset = VAD_SENSITIVITY_PRESETS[this.sensitivity] || VAD_SENSITIVITY_PRESETS[3];
    const speechThreshold = Math.max(preset.minFloor, this.noiseFloor * preset.mult);

    // Stream live level telemetry to UI for visual energy meter
    this.callbacks.onRmsUpdate?.(rms, speechThreshold, this.noiseFloor);

    if (rms > speechThreshold) {
      if (!this.isSpeaking) {
        // Speech onset
        this.isSpeaking = true;
        this.speechStartTime = now;
        this.silenceStartTime = 0;
        this.setState('SPEECH_DETECTED');
        this.callbacks.onSpeechStart?.();
      } else {
        // Voice continuation
        this.silenceStartTime = 0;
      }
    } else {
      if (this.isSpeaking) {
        // Voice offset or pause
        if (this.silenceStartTime === 0) {
          this.silenceStartTime = now;
        } else if (now - this.silenceStartTime >= this.silenceHangoverMs) {
          // Silence duration exceeded hangover
          const speechDuration = now - this.speechStartTime - (now - this.silenceStartTime);
          this.isSpeaking = false;
          this.silenceStartTime = 0;

          if (speechDuration >= this.minSpeechDurationMs) {
            this.callbacks.onSpeechEnd?.(speechDuration);
          } else {
            // Discard brief click/noise (< minSpeechDurationMs)
            this.setState('LISTENING');
          }
        }
      }
    }
  }

  public destroy() {
    this.stop();
  }
}
