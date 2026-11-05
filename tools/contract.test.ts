// (run inside a checkout of kfchem/meno, copied to src/lib/plugins/: each manifest of meno-plugins taken by Meno's own
//  reading of a manifest, acceptManifest, as Meno would take it from the catalogue - PLUGINS_DIR names its plugins/)
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { acceptManifest } from "./manifest";

const dir = process.env.PLUGINS_DIR ?? "";

describe("meno-plugins' manifests", () => {
  it("are each taken by Meno, of the id their folder is", () => {
    const ids = readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) {
      const m = acceptManifest(JSON.parse(readFileSync(join(dir, id, "manifest.json"), "utf8")));
      expect(m?.id, id).toBe(id);
    }
  });
});
