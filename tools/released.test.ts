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

  it("each plugin its list offers has a manifest it reads, and each manifest it cannot read is left out of its list", () => {
    // (the list it reads - catalogue.json, the entries the oldest released Meno reads whole: tools/build_catalogue.py -
    //  and each plugin here by its folder; a plugin of a role it does not know is offered only in a newer Meno's list)
    const read = acceptCatalogue(JSON.parse(readFileSync(catalogue, "utf8")));
    if ("why" in read) throw new Error(`the catalogue refused: ${read.why}`);
    const offered = new Set(read.catalogue.plugins.map((p) => p.id));
    const ids = readdirSync(dir, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of offered) expect(ids, `${id}: offered, with no folder here`).toContain(id);
    for (const id of ids) {
      const raw = JSON.parse(readFileSync(join(dir, id, "manifest.json"), "utf8")) as { contract?: number };
      // (of a contract it does not speak: shown as for a newer Meno, as it should be)
      if (!CONTRACTS.includes(raw.contract ?? 1)) continue;
      const reads = acceptManifest(raw)?.id === id;
      // (not read - a role it does not know, say: never in its list, so that such a plugin can never slip into it)
      if (!reads) expect(offered.has(id), `${id}: its manifest not read, yet offered in its list`).toBe(false);
      // (offered: read, of the id its folder is - a plugin it offers and cannot read would be one it lists and never adds)
      if (offered.has(id)) expect(reads, `${id}: offered in its list, its manifest not read`).toBe(true);
    }
  });
});
