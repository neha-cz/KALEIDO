import { forwardRef, useCallback } from "react";
import Spline from "@splinetool/react-spline";
import styles from "./CubeSection.module.css";

/** Hosted Spline scene (react-spline; Vite uses `@splinetool/react-spline`, not `/next`). */
const SPLINE_SCENE_URL =
  "https://prod.spline.design/Bqs5zZC-Z72lgF1A/scene.splinecode";

export const CubeSection = forwardRef(function CubeSection(_props, forwardedRef) {
  const setSectionRef = useCallback(
    (node) => {
      if (typeof forwardedRef === "function") forwardedRef(node);
      else if (forwardedRef) forwardedRef.current = node;
    },
    [forwardedRef]
  );

  return (
    <section
      ref={setSectionRef}
      className={styles.section}
      aria-labelledby="cube-section-heading"
    >
      <div className={styles.sectionIntro}>
        <h2 id="cube-section-heading" className={styles.subtitle}>
        Steering the model's voice from inside.
        </h2>
      </div>
      <div className={styles.sticky}>
        <div className={styles.splitRow}>
          <div className={styles.splineWrap}>
            <Spline scene={SPLINE_SCENE_URL} className={styles.splineCanvas} />
          </div>
          <div className={styles.copyColumn}>
            <p className={styles.lede}>
            A direction in activation space is added to the residual stream at every decode step, pulling the voice toward ego dissolution at the computational level.
            </p>
            <p className={styles.copyGradient}>
              Every token generated reflects genuine shifts in reasoning.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
});
