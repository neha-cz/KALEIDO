import { motion, useMotionValue, useScroll, useTransform } from "framer-motion";
import { useEffect, useRef } from "react";
import { usePrefersReducedMotion } from "../hooks/usePrefersReducedMotion.js";
import { finalBkgEzgifFramePath } from "../lib/ezgifPaths.js";
import { publicUrl } from "../lib/publicUrl.js";
import {
  prefetchFrameRing,
  safeDecodeImage,
  scheduleIdleFilmstripPrefetch,
} from "../lib/safeImageDecode.js";
import styles from "./FinalBkgAnimation.module.css";

const FRAME_COUNT = 240;
const FPS = 24;

const frameUrl = (index) => finalBkgEzgifFramePath(index);

export function FinalBkgAnimation() {
  const sectionRef = useRef(null);
  const img0Ref = useRef(null);
  const img1Ref = useRef(null);
  const reduce = usePrefersReducedMotion();
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "start start"],
  });
  const playbackProgress = useMotionValue(0);
  const animOpacity = useTransform(scrollYProgress, [0, 0.85], [0, 1]);
  const bgOpacity = useTransform(scrollYProgress, [0, 0.85], [1, 0]);
  const titleOpacity = useTransform(
    playbackProgress,
    [0, 0.74, 0.92, 1],
    [1, 1, 0.35, 0]
  );
  const ctaOpacity = useTransform(playbackProgress, [0, 0.965, 1], [0, 0, 1]);

  useEffect(() => {
    const sectionEl = sectionRef.current;
    const el0 = img0Ref.current;
    const el1 = img1Ref.current;
    if (!sectionEl || !el0 || !el1) return;

    let alive = true;
    let rafId = 0;

    const applyLayerClasses = (front0Visible) => {
      if (front0Visible) {
        el0.classList.add(styles.layerOn);
        el0.classList.remove(styles.layerOff);
        el1.classList.add(styles.layerOff);
        el1.classList.remove(styles.layerOn);
      } else {
        el1.classList.add(styles.layerOn);
        el1.classList.remove(styles.layerOff);
        el0.classList.add(styles.layerOff);
        el0.classList.remove(styles.layerOn);
      }
    };

    let frontIs0 = true;
    let displayed = 0;
    let t0 = performance.now();
    let wasIntersecting = false;

    const hiddenEl = () => (frontIs0 ? el1 : el0);

    const frameAtTime = (now) =>
      Math.min(
        FRAME_COUNT - 1,
        Math.floor(((now - t0) / 1000) * FPS)
      );

    const resetPlayback = () => {
      t0 = performance.now();
      frontIs0 = true;
      displayed = 0;
      playbackProgress.set(0);
      const u0 = frameUrl(0);
      el0.src = u0;
      el0.dataset.frameSrc = u0;
      el1.dataset.frameSrc = "";
      applyLayerClasses(true);
    };

    resetPlayback();
    prefetchFrameRing(frameUrl, 0, FRAME_COUNT, 72);
    let cancelIdlePrefetch = () => {};
    const idleDelayHandle = window.setTimeout(() => {
      cancelIdlePrefetch = scheduleIdleFilmstripPrefetch(
        frameUrl,
        FRAME_COUNT,
        { batchSize: 22 }
      );
    }, 2400);

    let pumping = false;

    const catchUp = async () => {
      while (alive) {
        const want = frameAtTime(performance.now());
        if (want === displayed) break;

        const hid = hiddenEl();
        const next = frameUrl(want);
        hid.fetchPriority = "high";
        hid.src = next;
        hid.dataset.frameSrc = next;
        const ok = await safeDecodeImage(hid, next);
        if (!alive) return;
        if (!ok) continue;

        frontIs0 = !frontIs0;
        applyLayerClasses(frontIs0);
        displayed = want;
        playbackProgress.set(want / (FRAME_COUNT - 1));
        prefetchFrameRing(frameUrl, displayed, FRAME_COUNT, 64);
      }
    };

    const tick = () => {
      if (!alive) return;
      if (!pumping) {
        pumping = true;
        void (async () => {
          try {
            await catchUp();
          } finally {
            pumping = false;
          }
        })();
      }
      rafId = requestAnimationFrame(tick);
    };

    const io = new IntersectionObserver(
      ([entry]) => {
        const nowIntersecting = entry.isIntersecting;
        if (nowIntersecting && !wasIntersecting) {
          // Replay when the user scrolls away and later comes back.
          resetPlayback();
        }
        wasIntersecting = nowIntersecting;
      },
      { threshold: 0.06 }
    );
    io.observe(sectionEl);

    rafId = requestAnimationFrame(tick);

    return () => {
      alive = false;
      clearTimeout(idleDelayHandle);
      cancelIdlePrefetch();
      io.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, [playbackProgress]);

  return (
    <section
      ref={sectionRef}
      className={styles.section}
      aria-labelledby="final-bkg-heading"
    >
      <motion.img
        src={publicUrl("cross-fade.png")}
        className={styles.baseBg}
        style={{ opacity: reduce ? 0 : bgOpacity }}
        alt=""
        decoding="async"
        draggable={false}
      />
      <motion.div
        className={styles.animationLayer}
        style={{ opacity: reduce ? 1 : animOpacity }}
      >
        <img
        ref={img0Ref}
        className={`${styles.layer} ${styles.layerOn}`}
        alt=""
        loading="eager"
        decoding="auto"
        fetchPriority="high"
        draggable={false}
        />
        <img
        ref={img1Ref}
        className={`${styles.layer} ${styles.layerOff}`}
        alt=""
        loading="eager"
        decoding="auto"
        fetchPriority="low"
        draggable={false}
        />
      </motion.div>
      <div className={styles.intro}>
        <motion.h2
          id="final-bkg-heading"
          className={styles.title}
          style={{ opacity: titleOpacity }}
        >
          Ready to dose our model?
        </motion.h2>
      </div>
      <motion.div
        className={styles.finalCtaWrap}
        style={{ opacity: ctaOpacity, pointerEvents: "none" }}
      >
        <motion.button
          type="button"
          className={styles.finalCta}
          initial={false}
          whileHover={
            reduce
              ? { scale: 1.02 }
              : {
                  scale: 1.03,
                  boxShadow: "0 6px 0 rgba(26, 43, 80, 0.2)",
                  transition: { duration: 0.2 },
                }
          }
          whileTap={{ scale: 0.96 }}
          style={{ pointerEvents: "auto" }}
          onClick={() =>
            window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" })
          }
        >
          <span className={styles.finalCtaBrackets} aria-hidden>
            <span className={`${styles.finalCtaCorner} ${styles.finalCtaCornerTl}`} />
            <span className={`${styles.finalCtaCorner} ${styles.finalCtaCornerTr}`} />
            <span className={`${styles.finalCtaCorner} ${styles.finalCtaCornerBl}`} />
            <span className={`${styles.finalCtaCorner} ${styles.finalCtaCornerBr}`} />
          </span>
          <span className={styles.finalCtaFiligree} aria-hidden>
            <svg
              className={styles.finalCtaFiligreeSvg}
              viewBox="0 0 140 72"
              preserveAspectRatio="xMinYMax meet"
            >
              <path
                d="M4 70 C 28 58, 20 32, 44 18 C 58 6, 62 2, 86 0 M 0 64 C 22 52, 18 30, 38 20 M 12 70 C 34 60, 30 40, 52 28 C 64 20, 72 12, 90 4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.35"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <svg
              className={`${styles.finalCtaFiligreeSvg} ${styles.finalCtaFiligreeRight}`}
              viewBox="0 0 140 72"
              preserveAspectRatio="xMaxYMax meet"
            >
              <path
                d="M4 70 C 28 58, 20 32, 44 18 C 58 6, 62 2, 86 0 M 0 64 C 22 52, 18 30, 38 20 M 12 70 C 34 60, 30 40, 52 28 C 64 20, 72 12, 90 4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.35"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span className={styles.finalCtaLabel}>BEGIN THE TRIP</span>
        </motion.button>
      </motion.div>
    </section>
  );
}
