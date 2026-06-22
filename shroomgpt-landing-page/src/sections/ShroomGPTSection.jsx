import { AutoplayFrameFilm } from "../components/AutoplayFrameFilm.jsx";
import { Container } from "../components/uimax/Container.jsx";
import styles from "./ShroomGPTSection.module.css";

const SUBTITLE_TEXT = "We flatten the energy landscape of thought.";

const LEDE_BEFORE = `Psychedelics, in the Entropic Brain framework, shallow the brain's cognitive energy landscape, dissolving the sharp basins that lock thought into familiar attractors and letting the mind roam between associations it would normally never connect. Under the modern Hopfield interpretation of attention, each transformer layer settles into minima of exactly this kind of energy landscape, governed by an inverse temperature β that controls how deep and separated those basins are. ShroomGPT lowers β directly inside the attention mechanism, `;

const LEDE_AFTER = `.`;

const LEDE_BELOW_FILM_BEFORE =
  "The result is a model that fundamentally thinks in a new way, mirroring how a psychedelic experience ";

const LEDE_BELOW_FILM_AFTER =
  ". In doing so, KALEIDO isn't just a prompt engineering framework. It is a model that reasons from a place most LLMs cannot reach.";

export function ShroomGPTSection() {
  return (
    <section
      className={styles.section}
      aria-labelledby="shroomgpt-heading"
    >
      <Container wide>
        <div className={styles.copy}>
          <p className={styles.label}>01 / KALEIDO</p>

          <h2
            id="shroomgpt-heading"
            className={styles.titleWrap}
            aria-label="ShroomGPT"
            style={{ margin: 0 }}
            aria-describedby="shroomgpt-subtitle"
          >
            <span className={styles.title}>Kaleido</span>
          </h2>

          <p id="shroomgpt-subtitle" className={styles.subtitle}>
            {SUBTITLE_TEXT}
          </p>

          <p className={styles.lede}>
            {LEDE_BEFORE}
            <strong className={styles.ledeEmphasis}>
            flattening the landscape so the model drifts between patterns instead of committing to one
            </strong>
            {LEDE_AFTER}
          </p>

          <AutoplayFrameFilm className={styles.film} />

          <p className={`${styles.lede} ${styles.ledeBelowFilm}`}>
            {LEDE_BELOW_FILM_BEFORE}
            <strong className={styles.ledeEmphasis}>
              unlocks your creative potential
            </strong>
            {LEDE_BELOW_FILM_AFTER}
          </p>
        </div>
      </Container>
    </section>
  );
}
