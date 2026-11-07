// (run inside a checkout of a RELEASED kfchem/meno - a tag of tools/released.txt - copied to src/lib/plugins/: the catalogue
//  built from these plugins (CATALOGUE names it) and each of their manifests (PLUGINS_DIR names plugins/), as that Meno
//  reads them. A Meno out already cannot be changed: a catalogue it refuses would take every plugin from it; a manifest of
//  a contract it speaks that it refuses would be a plugin it offers and cannot read. One of a contract it does not speak
//  is shown as for a newer Meno, as it should be)
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { acceptCatalogue, offersOf } from "./catalogue";
import { acceptManifest, CONTRACTS } from "./manifest";

const dir = process.env.PLUGINS_DIR ?? "";
const catalogue = process.env.CATALOGUE ?? "";

describe("these plugins, as a released Meno reads them", () => {
  it("the catalogue built from them is read whole", () => {
    const read = acceptCatalogue(JSON.parse(readFileSync(catalogue, "utf8")));
    expect("why" in read ? read.why : "", "the catalogue refused").toBe("");
    if ("why" in read) return;
    expect(read.catalogue.plugins.length).toBeGreaterThan(0);
    // (each offered - or, of a contract it does not speak, said to be for a newer Meno - never dropped)
    expect(offersOf(read.catalogue, new Map(), null).length).toBe(read.catalogue.plugins.length);
  });

  it("each manifest of a contract it speaks is read, of the id its folder is", () => {
    const ids = readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) {
      const raw = JSON.parse(readFileSync(join(dir, id, "manifest.json"), "utf8")) as { contract?: number };
      if (!CONTRACTS.includes(raw.contract ?? 1)) continue;
      expect(acceptManifest(raw)?.id, id).toBe(id);
    }
  });
});
