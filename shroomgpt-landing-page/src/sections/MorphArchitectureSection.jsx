import Spline from "@splinetool/react-spline";
import styles from "./MorphArchitectureSection.module.css";

const SPLINE_SCENE_URL =
  "https://prod.spline.design/Dvr9qtF09XUjVDxj/scene.splinecode";

export function MorphArchitectureSection() {
  return (
    <section
      className={styles.section}
      aria-labelledby="morph-architecture-heading"
    >
      <div className={styles.intro}>
        <h2 id="morph-architecture-heading" className={styles.title}>
        Dreaming through the model's vision.
        </h2>
      </div>
      <div className={styles.body}>
        <div className={styles.splitRow}>
          <div className={styles.copyColumn}>
            <p className={styles.lede}>
            We gradient-ascend the model's own visual feature geometry to generate dream vectors: surreal representations built directly from how the model sees. At every token, a fresh dream is injected into the residual stream, flooding the language layers with a shifting, churning visual prior.
            </p>
            <p className={styles.copyGradient}>
            The model doesn't describe a hallucination. It reasons through one.
            </p>
          </div>
          <div className={styles.splineWrap}>
            <Spline scene={SPLINE_SCENE_URL} className={styles.splineCanvas} />
          </div>
        </div>
      </div>
    </section>
  );
}
