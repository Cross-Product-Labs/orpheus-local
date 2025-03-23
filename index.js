// AudioWorklet PCM Player implementation
class AudioWorkletPCMPlayer {
  constructor(options = {}) {
    // Default options – ensure the sampleRate matches your PCM data
    this.options = Object.assign(
      {
        sampleRate: 24000,
        channels: 1,
        noiseGateThreshold: 0.001,
        noiseGateEnabled: true
      },
      options
    );
    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.options.sampleRate,
    });
    this.gainNode = this.audioCtx.createGain();
    this.gainNode.gain.value = 1;
    this.gainNode.connect(this.audioCtx.destination);
    this.initialized = false;
    this._initWorklet();
  }

  async _initWorklet() {
    // Inline AudioWorklet processor code
    const processorCode = `
  class PCMPlayerProcessor extends AudioWorkletProcessor {
    constructor() {
    super();
  // Internal buffer to hold incoming samples
  this.buffer = [];
  // Noise gate parameters
  this.noiseGateThreshold = 0.001; // Default threshold (adjustable)
  this.noiseGateEnabled = true;   // Enable by default
              // Listen for incoming PCM data messages
              this.port.onmessage = (event) => {
                if (event.data && event.data.samples) {
    // Append samples (assumed to be an array of float values in [-1,1])
    this.buffer.push(...event.data.samples);
                } else if (event.data && event.data.command === 'setNoiseGate') {
    // Update noise gate settings
    this.noiseGateThreshold = event.data.threshold;
  this.noiseGateEnabled = event.data.enabled;
                }
              };
            }
  process(inputs, outputs, parameters) {
              const output = outputs[0];
  // Process the first channel (assuming mono playback)
  const channel = output[0];
  const blockSize = channel.length;
  // Fill the output with samples from our buffer; pad with zeros if needed
  for (let i = 0; i < blockSize; i++) {
    let sample = this.buffer.length ? this.buffer.shift() : 0;

  // Apply noise gate if enabled
  if (this.noiseGateEnabled && Math.abs(sample) < this.noiseGateThreshold) {
    sample = 0;
                }

  channel[i] = sample;
              }
  return true;
            }
          }
  registerProcessor('pcm-player-processor', PCMPlayerProcessor);
  `;
    const blob = new Blob([processorCode], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    await this.audioCtx.audioWorklet.addModule(url);
    this.node = new AudioWorkletNode(this.audioCtx, "pcm-player-processor");
    this.node.connect(this.gainNode);
    this.initialized = true;
  }

  /**
   * Feed PCM data into the player.
   * Supports Int16Array, Uint8Array, Float32Array, Array, ArrayBuffer, and Buffer.
   */
  feed(data) {
    let floatData;
    if (data instanceof Int16Array) {
      // Convert 16-bit PCM to Float32 values in [-1,1]
      floatData = new Float32Array(data.length);
      for (let i = 0; i < data.length; i++) {
        floatData[i] = data[i] / 32768;
      }
    } else if (data instanceof Uint8Array) {
      // Convert Uint8Array (raw bytes) to Int16Array, then to Float32Array
      if (data.byteLength % 2 !== 0) {
        console.warn("Uint8Array length not even. Cannot interpret as 16-bit PCM.");
        return;
      }
      const int16Data = new Int16Array(data.buffer, data.byteOffset, data.byteLength / 2);
      floatData = new Float32Array(int16Data.length);
      for (let i = 0; i < int16Data.length; i++) {
        floatData[i] = int16Data[i] / 32768;
      }
    } else if (data instanceof Float32Array) {
      floatData = data;
    } else if (Array.isArray(data)) {
      floatData = new Float32Array(data);
    } else if (data instanceof ArrayBuffer) {
      const int16Data = new Int16Array(data);
      floatData = new Float32Array(int16Data.length);
      for (let i = 0; i < int16Data.length; i++) {
        floatData[i] = int16Data[i] / 32768;
      }
    } else if (typeof Buffer !== 'undefined' && data instanceof Buffer) {
      const int16Data = new Int16Array(data.buffer, data.byteOffset, data.byteLength / 2);
      floatData = new Float32Array(int16Data.length);
      for (let i = 0; i < int16Data.length; i++) {
        floatData[i] = int16Data[i] / 32768;
      }
    } else {
      console.warn("Unsupported data type passed to feed().");
      return;
    }
    if (this.initialized) {
      // Post the float samples to the worklet; using Array.from for structured cloning
      this.node.port.postMessage({ samples: Array.from(floatData) });
    } else {
      console.warn("AudioWorklet not yet initialized.");
    }
  }

  /**
   * Set playback volume.
   * @param {number} value - Volume value (0.0 to 1.0)
*/
  volume(value) {
    if (this.gainNode) {
      this.gainNode.gain.value = value;
    }
  }

  /**
   * Set noise gate threshold.
   * @param {number} threshold - Threshold value (0.0 to 1.0)
* @param {boolean} enabled - Whether the noise gate is enabled
*/
  setNoiseGate(threshold, enabled) {
    if (this.initialized && this.node) {
      this.node.port.postMessage({
        command: 'setNoiseGate',
        threshold: threshold,
        enabled: enabled
      });
    }
  }

  /**
   * Clean up and shut down the player.
   */
  destroy() {
    if (this.node) {
      this.node.disconnect();
    }
    if (this.gainNode) {
      this.gainNode.disconnect();
    }
    if (this.audioCtx && this.audioCtx.close) {
      this.audioCtx.close();
    }
  }
}

// Setup function for the player
function setupPlayer() {
  return new AudioWorkletPCMPlayer({
    sampleRate: 24000,
    noiseGateThreshold: noiseGateThreshold,
    noiseGateEnabled: noiseGateEnabled
  });
}

const playButton = document.getElementById('playButton');
const statusDiv = document.getElementById('status');
const timingInfoDiv = document.getElementById('timingInfo');
const errorDiv = document.getElementById('error');
const textInput = document.getElementById('textInput');
const bufferSizeInput = document.getElementById('bufferSizeInput');
const bufferSizeSlider = document.getElementById('bufferSizeSlider');
const bufferInfoSpan = document.getElementById('bufferInfo');
const audioContainer = document.getElementById('audioContainer');
const audioInfo = document.getElementById('audioInfo');
const audioPlayer = document.getElementById('audioPlayer');
const downloadButton = document.getElementById('downloadButton');

let player = null;
let isSpeaking = false;
let audioChunks = []; // Store all audio chunks
let audioDuration = 0; // Track audio duration

// Default buffer size in KB - reduce for lower latency
const DEFAULT_BUFFER_SIZE_KB = 8;  // Changed from 4 to 8
// Calculate buffer size in bytes (KB * 1024)
let bufferSizeKB = DEFAULT_BUFFER_SIZE_KB;
let MIN_BUFFER_BYTES = bufferSizeKB * 1024;

// Default noise gate settings
const DEFAULT_NOISE_GATE_THRESHOLD = 0.001;
const DEFAULT_NOISE_GATE_ENABLED = true;

let noiseGateThreshold = DEFAULT_NOISE_GATE_THRESHOLD;
let noiseGateEnabled = DEFAULT_NOISE_GATE_ENABLED;

// Default generation parameters
const DEFAULT_TEMPERATURE = 0.6;
const DEFAULT_TOP_P = 0.9;
const DEFAULT_REP_PENALTY = 1.1;
const DEFAULT_MAX_TOKENS = 4096;

// Current generation parameters
let temperature = DEFAULT_TEMPERATURE;
let topP = DEFAULT_TOP_P;
let repPenalty = DEFAULT_REP_PENALTY;
let maxTokens = DEFAULT_MAX_TOKENS;

// Add these variables at the top with other globals
let audioContext;
let analyser;
let spectrogramCanvas;
let spectrogramCtx;
let isDrawing = false;

// Add these variables to track average latencies
let totalGenerations = 0;
let totalFirstChunkLatency = 0;
let totalPlaybackStartLatency = 0;
let averageFirstChunkLatency = 0;
let averagePlaybackStartLatency = 0;

// Function to update buffer size
function updateBufferSize(sizeKB) {
  bufferSizeKB = parseInt(sizeKB) || DEFAULT_BUFFER_SIZE_KB;
  MIN_BUFFER_BYTES = bufferSizeKB * 1024;
  bufferInfoSpan.textContent = `(${bufferSizeKB} KB = ${MIN_BUFFER_BYTES} bytes)`;

  // Sync slider and input if they don't match
  bufferSizeInput.value = bufferSizeKB;
  bufferSizeSlider.value = bufferSizeKB;
}

// Function to reset buffer size to default
function resetBufferSize() {
  updateBufferSize(DEFAULT_BUFFER_SIZE_KB);
}

// Initialize with default value
updateBufferSize(DEFAULT_BUFFER_SIZE_KB);

// Function to update temperature
function updateTemperature(value) {
  temperature = parseFloat(value) || DEFAULT_TEMPERATURE;

  // Sync slider and input if they don't match
  document.getElementById('temperatureInput').value = temperature;
  document.getElementById('temperatureSlider').value = temperature;
}

// Function to reset temperature to default
function resetTemperature() {
  updateTemperature(DEFAULT_TEMPERATURE);
}

// Function to update top_p
function updateTopP(value) {
  topP = parseFloat(value) || DEFAULT_TOP_P;

  // Sync slider and input if they don't match
  document.getElementById('topPInput').value = topP;
  document.getElementById('topPSlider').value = topP;
}

// Function to reset top_p to default
function resetTopP() {
  updateTopP(DEFAULT_TOP_P);
}

// Function to update repetition penalty
function updateRepPenalty(value) {
  repPenalty = parseFloat(value) || DEFAULT_REP_PENALTY;

  // Sync slider and input if they don't match
  document.getElementById('repPenaltyInput').value = repPenalty;
  document.getElementById('repPenaltySlider').value = repPenalty;
}

// Function to reset repetition penalty to default
function resetRepPenalty() {
  updateRepPenalty(DEFAULT_REP_PENALTY);
}

// Function to update max tokens
function updateMaxTokens(value) {
  maxTokens = parseInt(value) || DEFAULT_MAX_TOKENS;

  // Sync slider and input if they don't match
  document.getElementById('maxTokensInput').value = maxTokens;
  document.getElementById('maxTokensSlider').value = maxTokens;
}

// Function to reset max tokens to default
function resetMaxTokens() {
  updateMaxTokens(DEFAULT_MAX_TOKENS);
}

// Initialize with default values
function initializeSettings() {
  updateBufferSize(DEFAULT_BUFFER_SIZE_KB);

  // Set initial generation parameter values
  updateTemperature(DEFAULT_TEMPERATURE);
  updateTopP(DEFAULT_TOP_P);
  updateRepPenalty(DEFAULT_REP_PENALTY);
  updateMaxTokens(DEFAULT_MAX_TOKENS);
}

// Call initialization on page load
window.addEventListener('DOMContentLoaded', initializeSettings);

// Initialize audio on first user interaction
document.body.addEventListener('click', function () {
  if (!player) {
    player = setupPlayer();
  }
}, { once: true });

async function handlePlay() {
  try {
    playButton.disabled = true;
    statusDiv.textContent = 'Generating audio...';
    timingInfoDiv.textContent = '';
    errorDiv.textContent = '';
    audioContainer.style.display = 'none';

    // Reset audio data
    audioChunks = [];
    audioDuration = 0;

    const text = textInput.value.trim();
    if (!text) {
      throw new Error('Please enter some text');
    }

    // Get the selected voice
    const voice = document.getElementById('voiceSelect').value;

    // Initialize player if not already done
    if (!player) {
      player = setupPlayer();
    } else {
      // Update noise gate settings in case they've changed
      player.setNoiseGate(noiseGateThreshold, noiseGateEnabled);
    }

    // Record start time when button is clicked
    const startTime = performance.now();

    // Create event source for streaming with all parameters
    const params = new URLSearchParams({
      text: text,
      voice: voice,
      temperature: temperature,
      top_p: topP,
      repetition_penalty: repPenalty,
      max_tokens: maxTokens
    });

    const eventSource = new EventSource(`http://localhost:8000/stream-audio?${params.toString()}`);

    // Buffer settings
    const initialBuffer = [];
    let isBuffering = true;
    let totalBufferedBytes = 0;
    isSpeaking = true;
    let firstChunkReceived = false;
    let startPlayTime = 0;

    eventSource.addEventListener('format', (e) => {
      const format = JSON.parse(e.data);
      console.log('Audio format:', format);
    });

    eventSource.addEventListener('audio', async (e) => {
      // Calculate latency on first chunk
      if (!firstChunkReceived) {
        const firstChunkLatency = performance.now() - startTime;
        console.log(`First audio chunk received after ${firstChunkLatency.toFixed(0)}ms`);
        statusDiv.textContent = `First chunk received after ${firstChunkLatency.toFixed(0)}ms`;
        firstChunkReceived = true;

        // Update first chunk latency averages
        totalFirstChunkLatency += firstChunkLatency;
        totalGenerations++;
        averageFirstChunkLatency = totalFirstChunkLatency / totalGenerations;

        // Store firstChunkLatency in a variable accessible throughout the function
        this.firstChunkLatency = firstChunkLatency;
      }

      // Make sure e.data exists and is a string before trying to match
      if (e.data && typeof e.data === 'string') {
        // Convert hex string back to audio data
        const matches = e.data.match(/.{1,2}/g);
        if (matches) {
          const chunk = new Uint8Array(matches.map(byte => parseInt(byte, 16)));

          // Store all chunks for replay
          audioChunks.push(chunk);

          // Calculate approximate duration (assuming 24kHz, 16-bit mono)
          // Each sample is 2 bytes, so bytes/2 gives us sample count
          // Then divide by sample rate to get seconds
          audioDuration += chunk.length / 2 / 24000;

          // Buffer initial chunks
          if (isBuffering) {
            initialBuffer.push(chunk);
            totalBufferedBytes += chunk.length;
            statusDiv.textContent = `Buffering audio... (${Math.round(totalBufferedBytes / 1024)} KB)`;

            // Start playback when we have enough data or after a timeout
            const bufferTimeoutMs = 1000; // 1 second max wait
            const hasEnoughData = totalBufferedBytes >= MIN_BUFFER_BYTES;
            const hasWaitedTooLong = performance.now() - startTime > bufferTimeoutMs;

            if (hasEnoughData || (hasWaitedTooLong && totalBufferedBytes > 0)) {
              isBuffering = false;
              startPlayTime = performance.now();
              const playbackStartLatency = startPlayTime - startTime;

              // Update playback start latency averages
              totalPlaybackStartLatency += playbackStartLatency;
              averagePlaybackStartLatency = totalPlaybackStartLatency / totalGenerations;

              statusDiv.innerHTML = 'Playing audio...';
              // Display current generation metrics on one line
              timingInfoDiv.innerHTML = `First chunk: ${this.firstChunkLatency ? this.firstChunkLatency.toFixed(0) : 'N/A'}ms | Playback start: ${playbackStartLatency.toFixed(0)}ms`;
              // Display average metrics on a separate line
              timingInfoDiv.innerHTML += `<br>Average (${totalGenerations} generations): First chunk: ${averageFirstChunkLatency.toFixed(0)}ms | Playback start: ${averagePlaybackStartLatency.toFixed(0)}ms`;

              // Feed the initial buffer to the player
              for (const bufferedChunk of initialBuffer) {
                if (player) {
                  player.feed(bufferedChunk);
                }
              }
            }
            return;
          }

          // Feed PCM data to player after buffering phase
          if (player) {
            player.feed(chunk);
          }
        } else {
          console.warn('Invalid hex data received:', e.data);
        }
      } else {
        console.warn('Invalid data received from event source');
      }
    });

    eventSource.onerror = () => {
      eventSource.close();
      playButton.disabled = false;
      isSpeaking = false;

      if (isBuffering) {
        errorDiv.textContent = 'Error streaming audio';
        statusDiv.textContent = '';
      } else {
        const playbackTime = ((performance.now() - startPlayTime) / 1000).toFixed(2);
        statusDiv.textContent = 'Playback finished';

        // Show audio info and controls
        audioInfo.textContent = `Audio duration: ${audioDuration.toFixed(2)}s | Playback time: ${playbackTime}s`;

        // Create WAV file from the audio chunks
        createAndSetAudioElement();

        audioContainer.style.display = 'block';

        // Don't clear timing info
        setTimeout(() => {
          statusDiv.textContent = '';
        }, 3000);
      }
    };

  } catch (error) {
    errorDiv.textContent = `Error: ${error.message}`;
    statusDiv.textContent = '';
    playButton.disabled = false;
    console.error('Error:', error);
  }
}

// Add this function before createAndSetAudioElement
function setupSpectrogram() {
  spectrogramCanvas = document.getElementById('spectrogram');
  spectrogramCtx = spectrogramCanvas.getContext('2d');

  // Set canvas size
  spectrogramCanvas.width = spectrogramCanvas.offsetWidth || 500;
  spectrogramCanvas.height = spectrogramCanvas.offsetHeight || 200;

  // Clear canvas initially
  spectrogramCtx.fillStyle = '#2a2a2a';
  spectrogramCtx.fillRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);

  // Wait for audio to be loaded
  audioPlayer.addEventListener('loadedmetadata', function () {
    try {
      // Fetch the audio data
      fetch(audioPlayer.src)
        .then(response => response.arrayBuffer())
        .then(arrayBuffer => {
          // Convert audio data to PCM samples
          const pcmData = new Int16Array(arrayBuffer.slice(44)); // Skip WAV header

          // Calculate how many samples to combine for each pixel
          const samplesPerPixel = Math.ceil(pcmData.length / spectrogramCanvas.width);

          // Draw waveform
          spectrogramCtx.strokeStyle = '#4CAF50';
          spectrogramCtx.lineWidth = 1;
          spectrogramCtx.beginPath();

          for (let x = 0; x < spectrogramCanvas.width; x++) {
            // Get min and max values for this pixel's sample range
            const startSample = x * samplesPerPixel;
            const endSample = Math.min(startSample + samplesPerPixel, pcmData.length);

            let min = 32767;
            let max = -32768;

            for (let i = startSample; i < endSample; i++) {
              const sample = pcmData[i];
              min = Math.min(min, sample);
              max = Math.max(max, sample);
            }

            // Convert to canvas coordinates
            const minY = ((min / 32768) + 1) / 2 * spectrogramCanvas.height;
            const maxY = ((max / 32768) + 1) / 2 * spectrogramCanvas.height;

            // Draw vertical line from min to max
            spectrogramCtx.moveTo(x + 0.5, minY);
            spectrogramCtx.lineTo(x + 0.5, maxY);
          }

          spectrogramCtx.stroke();

          // Draw chunk boundaries
          let currentSample = 0;
          spectrogramCtx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          spectrogramCtx.lineWidth = 1;

          audioChunks.forEach((chunk, index) => {
            // Calculate x position for this chunk boundary
            const chunkSamples = chunk.length / 2; // 2 bytes per sample
            currentSample += chunkSamples;
            const x = (currentSample / pcmData.length) * spectrogramCanvas.width;

            // Draw vertical line for chunk boundary
            spectrogramCtx.beginPath();
            spectrogramCtx.moveTo(x, 0);
            spectrogramCtx.lineTo(x, spectrogramCanvas.height);
            spectrogramCtx.stroke();
          });

          // Draw legend
          spectrogramCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
          spectrogramCtx.font = '12px Arial';
          spectrogramCtx.textAlign = 'right';
          spectrogramCtx.textBaseline = 'bottom';
          spectrogramCtx.fillText(`Total chunks: ${audioChunks.length}`, spectrogramCanvas.width - 5, spectrogramCanvas.height - 5);

        })
        .catch(error => {
          console.error('Error creating waveform:', error);
        });

    } catch (error) {
      console.error('Error setting up waveform:', error);
    }
  }, { once: true }); // Only run once
}

// Modify createAndSetAudioElement to call setupSpectrogram
function createAndSetAudioElement() {
  if (audioChunks.length === 0) {
    return;
  }

  // Combine all chunks into one array
  let totalLength = 0;
  audioChunks.forEach(chunk => {
    totalLength += chunk.length;
  });

  const combinedChunks = new Uint8Array(totalLength);
  let offset = 0;

  audioChunks.forEach(chunk => {
    combinedChunks.set(chunk, offset);
    offset += chunk.length;
  });

  // Convert to 16-bit PCM
  const pcmData = new Int16Array(combinedChunks.buffer, combinedChunks.byteOffset, combinedChunks.byteLength / 2);

  // Create WAV file
  const wavFile = createWavFile(pcmData, 24000);

  // Create blob and set as audio source
  const blob = new Blob([wavFile], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);

  // Set the audio element source
  audioPlayer.src = url;

  // Setup spectrogram after setting the source
  setupSpectrogram();

  // Clean up the URL when the page is unloaded
  window.addEventListener('unload', () => {
    URL.revokeObjectURL(url);
  });
}

// Function to download the audio as a WAV file
function downloadAudio() {
  if (audioChunks.length === 0) {
    errorDiv.textContent = 'No audio available to download';
    return;
  }

  // Use the current audio element's source if available
  if (audioPlayer.src) {
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = audioPlayer.src;
    a.download = 'tts-audio.wav';
    document.body.appendChild(a);
    a.click();

    // Clean up
    setTimeout(() => {
      document.body.removeChild(a);
    }, 100);
  } else {
    errorDiv.textContent = 'Audio source not available';
  }
}

// Function to create a WAV file from PCM data
function createWavFile(pcmData, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = pcmData.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // WAV header
  // "RIFF" chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // "fmt " sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // audio format (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // "data" sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write PCM data
  const offset = 44;
  for (let i = 0; i < pcmData.length; i++) {
    view.setInt16(offset + i * bytesPerSample, pcmData[i], true);
  }

  return buffer;
}

// Helper function to write a string to a DataView
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}