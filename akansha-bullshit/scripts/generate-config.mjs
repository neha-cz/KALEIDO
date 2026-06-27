import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");

const chatUrl =
  process.env.KALEIDO_CHAT_URL || "https://YOUR-USERNAME-kaleido.hf.space/chat";

const content = `window.KALEIDO_CONFIG = {
  chatUrl: ${JSON.stringify(chatUrl)},
};
`;

fs.writeFileSync(path.join(root, "config.js"), content);
console.log("Generated config.js with chatUrl:", chatUrl);
