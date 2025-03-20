async function playAudio() {
  try {
    const response = await fetch('http://localhost:8000/stream-audio');
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  } catch (error) {
    console.error('Error playing audio:', error);
  }
}