import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import styles from "./HeroSection.module.css";


const line1 = ["TAKE", "YOUR", "AI"];
const line2 = ["ON", "A", "TRIP."];

const LINE1_STR = "TAKE OUR AI";
const LINE2_STR = "ON A TRIP.";

/* Nested upper semicircles. Larger R = gentler; ARC_Y_FLATTEN <1 = flatter hump. */
const ARC_R1_EM = 2.2;
/* Same arc-length per letter gap as line 1: R1/(n1-1) = R2/(n2-1) → R2 = R1 * (n2-1)/(n1-1) */
const N_LINE1 = 11; /* "TAKE YOUR AI" */
const N_LINE2 = 10; /* "ON A TRIP." */
const ARC_R2_EM = ARC_R1_EM * (N_LINE2 - 1) / (N_LINE1 - 1);

/* 1 = true circle height; lower = flatter arch. */
const ARC_Y_FLATTEN = 0.62;

/* Shared span so both lines use the same angular packing strategy */
const ARC_ANGLE_T_MIN = 0.3;
const ARC_ANGLE_T_MAX = 0.7;

const SCATTER_END = 0.45;


function pseudoRandom(seed) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function line1WordIndexForCharAt(i) {
  if (i < 3) return 0;
  if (i < 4) return 0;
  if (i < 8) return 1;
  if (i < 9) return 1;
  return 2;
}

function line2WordIndexForCharAt(i) {
  if (i < 2) return 3;
  if (i < 3) return 3;
  if (i < 4) return 4;
  if (i < 5) return 4;
  return 5;
}

function Letter({ ch, index, progress, reduce, arc, arcEm, arcRotateDeg, delay }) {
  const dirX = pseudoRandom(index + 1) * 2 - 1;
  const dirY = pseudoRandom(index + 41) * 2 - 1;
  const driftX = useTransform(
    progress,
    [0, SCATTER_END],
    [0, dirX * 280]
  );
  const driftY = useTransform(
    progress,
    [0, SCATTER_END],
    [0, dirY * 210 - (90 + pseudoRandom(index + 77) * 70)]
  );
  const spin = useTransform(
    progress,
    [0, SCATTER_END],
    [0, (pseudoRandom(index + 101) * 2 - 1) * 46]
  );
  const fade = useTransform(
    progress,
    [0, SCATTER_END],
    [1, 0.08]
  );

  if (ch === " ") {
    if (arc && !reduce) {
      return (
        <span
          className={styles.arcLetter}
          style={{
            transform: `translate(${arcEm.x}em, ${arcEm.y}em) rotate(${arcRotateDeg}deg)`,
          }}
        >
          <span className={styles.space} aria-hidden />
        </span>
      );
    }
    return <span className={styles.space} aria-hidden>{ch}</span>;
  }
  if (reduce) {
    return <span className={styles.letter}>{ch}</span>;
  }

  if (arc) {
    return (
      <span
        className={styles.arcLetter}
        style={{
          transform: `translate(${arcEm.x}em, ${arcEm.y}em) rotate(${arcRotateDeg}deg)`,
        }}
      >
        <motion.span
          className={styles.letter}
          initial={{ opacity: 0, y: 24, filter: "blur(9px)" }}
          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
          transition={{
            duration: 0.72,
            delay,
            ease: [0.16, 1, 0.3, 1],
          }}
          style={{ x: driftX, y: driftY, rotate: spin, opacity: fade }}
        >
          {ch}
        </motion.span>
      </span>
    );
  }

  return (
    <motion.span
      className={styles.letter}
      style={{ x: driftX, y: driftY, rotate: spin, opacity: fade }}
    >
      {ch}
    </motion.span>
  );
}

function Word({ children, delay, reduce, progress, wordIndex }) {
  return (
    <motion.span
      className={styles.word}
      initial={{
        opacity: 0,
        y: reduce ? 10 : 28,
        filter: reduce ? "none" : "blur(10px)",
      }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{
        duration: reduce ? 0.35 : 0.75,
        ease: [0.16, 1, 0.3, 1],
        delay,
      }}
    >
      {Array.from(children).map((ch, i) => (
        <Letter
          key={`${ch}-${i}`}
          ch={ch}
          index={wordIndex * 24 + i}
          progress={progress}
          reduce={reduce}
        />
      ))}
    </motion.span>
  );
}

/**
 * One half-circle: θ from π → 0 (left → right). up=true: convex “rainbow” hump, center at the top of the arch.
 * Second line reuses the same hump with a smaller R so it nests inside the first (like small “THE” over a wide title).
 */
function ArcTextLine({ text, radiusEm, up, lineKey, baseDelay, step, progress, wordIndexForChar }) {
  const n = text.length;
  if (n < 1) return null;
  if (n === 1) {
    const t = 0.5;
    const theta = Math.PI * (ARC_ANGLE_T_MAX * (1 - t) + ARC_ANGLE_T_MIN * t);
    const x = radiusEm * Math.cos(theta);
    const y =
      (up ? -1 : 1) * radiusEm * Math.sin(theta) * ARC_Y_FLATTEN;
    const rotDeg = ((Math.PI / 2 - theta) * 180) / Math.PI;
    return (
      <span
        className={
          lineKey === 2 ? `${styles.arcRow} ${styles.arcRowInner}` : styles.arcRow
        }
        data-line={lineKey}
      >
        <Letter
          ch={text[0]}
          index={200 * lineKey}
          progress={progress}
          reduce={false}
          arc
          delay={baseDelay}
          arcEm={{ x, y }}
          arcRotateDeg={rotDeg}
        />
      </span>
    );
  }
  return (
    <span
      className={
        lineKey === 2 ? `${styles.arcRow} ${styles.arcRowInner}` : styles.arcRow
      }
      data-line={lineKey}
    >
      {Array.from(text).map((ch, i) => {
        const t = n > 1 ? i / (n - 1) : 0.5;
        const theta = Math.PI * (ARC_ANGLE_T_MAX * (1 - t) + ARC_ANGLE_T_MIN * t);
        const x = radiusEm * Math.cos(theta);
        const y =
          (up ? -1 : 1) * radiusEm * Math.sin(theta) * ARC_Y_FLATTEN;
        const rotDeg = ((Math.PI / 2 - theta) * 180) / Math.PI;
        const w = wordIndexForChar(i);
        return (
          <Letter
            key={`${ch}-${i}`}
            ch={ch}
            index={200 * lineKey + i}
            progress={progress}
            reduce={false}
            arc
            delay={baseDelay + w * step + i * 0.02}
            arcEm={{ x, y }}
            arcRotateDeg={rotDeg}
          />
        );
      })}
    </span>
  );
}

export function HeroSection({ scrollRootRef }) {
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({
    target: scrollRootRef,
    offset: ["start start", "end start"],
  });
  const ctaFade = useTransform(scrollYProgress, [0, 0.45], [1, 0]);

  const baseDelay = reduce ? 0.08 : 0.45;
  const step = reduce ? 0.04 : 0.09;

  const scrollToPage = () => {
    document.getElementById("main-content")?.scrollIntoView({
      behavior: reduce ? "auto" : "smooth",
    });
  };

  return (
    <section className={styles.section} aria-label="Hero">
      <motion.div className={styles.inner}>
        <h1 className={styles.headline}>
          <span className={styles.headlineLines}>
            {reduce ? (
              <>
                <span className={styles.wordRow}>
                  {line1.map((w, i) => (
                    <Word
                      key={w}
                      delay={baseDelay + i * step}
                      reduce={reduce}
                      progress={scrollYProgress}
                      wordIndex={i}
                    >
                      {w}
                    </Word>
                  ))}
                </span>
                <span className={styles.wordRow}>
                  {line2.map((w, j) => (
                    <Word
                      key={w}
                      delay={baseDelay + (line1.length + j) * step}
                      reduce={reduce}
                      progress={scrollYProgress}
                      wordIndex={line1.length + j}
                    >
                      {w}
                    </Word>
                  ))}
                </span>
              </>
            ) : (
              <>
                <ArcTextLine
                  text={LINE1_STR}
                  radiusEm={ARC_R1_EM}
                  up
                  lineKey={1}
                  baseDelay={baseDelay}
                  step={step}
                  progress={scrollYProgress}
                  wordIndexForChar={line1WordIndexForCharAt}
                />
                <ArcTextLine
                  text={LINE2_STR}
                  radiusEm={ARC_R2_EM}
                  up
                  lineKey={2}
                  baseDelay={baseDelay}
                  step={step}
                  progress={scrollYProgress}
                  wordIndexForChar={line2WordIndexForCharAt}
                />
              </>
            )}
          </span>
        </h1>

        <div className={styles.ctaWrap}>
          <motion.button
            type="button"
            className={styles.posterCta}
            onClick={scrollToPage}
            style={{ opacity: reduce ? 1 : ctaFade }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.6,
              delay: reduce ? 0.3 : 1.05,
              ease: [0.16, 1, 0.3, 1],
            }}
            whileHover={
              reduce
                ? { scale: 1.02 }
                : {
                    scale: 1.04,
                    boxShadow: "0 6px 0 rgba(26, 43, 80, 0.2)",
                    transition: { duration: 0.2 },
                  }
            }
            whileTap={{ scale: 0.98 }}
          >
            <span className={styles.posterCtaBrackets} aria-hidden>
              <span
                className={`${styles.posterCtaCorner} ${styles.posterCtaCornerTl}`}
              />
              <span
                className={`${styles.posterCtaCorner} ${styles.posterCtaCornerTr}`}
              />
              <span
                className={`${styles.posterCtaCorner} ${styles.posterCtaCornerBl}`}
              />
              <span
                className={`${styles.posterCtaCorner} ${styles.posterCtaCornerBr}`}
              />
            </span>
            <span className={styles.posterCtaFiligree} aria-hidden>
              <svg
                className={styles.posterCtaFiligreeSvg}
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
                className={`${styles.posterCtaFiligreeSvg} ${styles.posterCtaFiligreeRight}`}
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
            <span className={styles.posterCtaLabel}>BEGIN THE TRIP</span>
          </motion.button>
        </div>
      </motion.div>
    </section>
  );
}
