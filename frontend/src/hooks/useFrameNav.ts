import { useCallback, useEffect, useRef, useState } from "react";
import { useAppStore } from "@/stores/appStore";

export function useFrameNav() {
  const numFrames = useAppStore((s) => s.numFrames);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame);

  // Ref for interval callback (synchronous read), state for re-renders
  const playRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fpsRef = useRef(5);

  const next = useCallback(() => {
    setCurrentFrame(Math.min(currentFrame + 1, numFrames - 1));
  }, [currentFrame, numFrames, setCurrentFrame]);

  const prev = useCallback(() => {
    setCurrentFrame(Math.max(currentFrame - 1, 0));
  }, [currentFrame, setCurrentFrame]);

  const first = useCallback(() => setCurrentFrame(0), [setCurrentFrame]);
  const last = useCallback(
    () => setCurrentFrame(numFrames - 1),
    [numFrames, setCurrentFrame]
  );

  const goTo = useCallback(
    (f: number) => setCurrentFrame(Math.max(0, Math.min(f, numFrames - 1))),
    [numFrames, setCurrentFrame]
  );

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const pause = useCallback(() => {
    playRef.current = false;
    setIsPlaying(false);
    stopTimer();
  }, [stopTimer]);

  const play = useCallback(() => {
    if (playRef.current) return;

    // If at last frame, wrap around to beginning
    const state = useAppStore.getState();
    if (state.currentFrame >= state.numFrames - 1) {
      useAppStore.setState({ currentFrame: 0 });
    }

    playRef.current = true;
    setIsPlaying(true);
    timerRef.current = setInterval(() => {
      useAppStore.setState((s) => {
        const next = s.currentFrame + 1;
        if (next >= s.numFrames) {
          playRef.current = false;
          setIsPlaying(false);
          if (timerRef.current) clearInterval(timerRef.current);
          return {};
        }
        return { currentFrame: next };
      });
    }, 1000 / fpsRef.current);
  }, []);

  const setFps = useCallback((fps: number) => {
    fpsRef.current = fps;
    if (playRef.current) {
      stopTimer();
      playRef.current = false;
      setIsPlaying(false);
      // Restart with new FPS
      playRef.current = true;
      setIsPlaying(true);
      timerRef.current = setInterval(() => {
        useAppStore.setState((s) => {
          const next = s.currentFrame + 1;
          if (next >= s.numFrames) {
            playRef.current = false;
            setIsPlaying(false);
            if (timerRef.current) clearInterval(timerRef.current);
            return {};
          }
          return { currentFrame: next };
        });
      }, 1000 / fps);
    }
  }, [stopTimer]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === " ") {
        e.preventDefault();
        playRef.current ? pause() : play();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [next, prev, play, pause]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopTimer();
  }, [stopTimer]);

  return { currentFrame, numFrames, next, prev, first, last, goTo, play, pause, isPlaying, setFps };
}
