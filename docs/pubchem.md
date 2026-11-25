# PubChem: what the plugin relies on of PubChem's service

The PubChem plugin finds a structure by its name at PubChem, for Meno's
role *A structure found by its name, in a database outside Meno*: Quick
Add's field looks a name up there only when the person chooses that row,
and the first time Meno asks whether names may go to
pubchem.ncbi.nlm.nih.gov. What goes there is the name looked up, in the
body of one request, and nothing else: no header says who asks or from
where (`User-Agent: meno-plugin-pubchem/<version>`, `Accept:
application/json`, and what HTTP itself needs). It uses Python alone.

The plugin does no chemistry. It maps the fields PubChem gives to Meno's
and refuses, in words, an answer that does not fit them; Meno reads the
SMILES and compares it with the formula and charge given before anything
is drawn.

Every fact about the service it relies on is listed here with the page it
was read on (each page's own "Last updated" in brackets). A fact changed
at the source is a change to `plugins/pubchem/worker.py` and this page
together. Its tests (`tests/test_plugin_pubchem.py`) run against a
stand-in on the test's own computer, with answers written by hand; they
reach no live service.

| What the worker relies on | As it stands | Read on |
| --- | --- | --- |
| A name, to the ids of the compounds it names | `POST /rest/pug/compound/name/cids/JSON`, the body `name=<the name, form-encoded>` (a `+` in a name sent as `%2B`), `Content-Type: application/x-www-form-urlencoded`: the name is never in the address | PUG REST Tutorial (September 15, 2026), *How To Use HTTP POST* and *Special Characters in the URL*; PUG REST (September 15, 2026), *Request (POST) Body* |
| Which names match | the whole name, exactly - the default; `name_type` is not sent | PUG REST Tutorial, *By Name* |
| The ids, to the records | `GET /rest/pug/compound/cid/<ids>/property/Title,MolecularFormula,SMILES,Charge,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount/JSON`, the first ids in PubChem's order | PUG REST, *Compound Property Tables* |
| The SMILES | `SMILES`, which includes configurations and isotopes - never `ConnectivitySMILES`, which has neither | PUG REST, *Compound Property Tables* |
| The formula | Hill order, the total charge at its end; given to Meno as written | PUG REST, *Compound Property Tables* |
| The answers' shape | `IdentifierList.CID`; `PropertyTable.Properties`, each with its `CID`; a fault's `Fault.Code`. Each property is optional in the schema: a record without one the plugin gives is refused, never filled in | the schema PUG REST links (*Schemas*): `https://pubchem.ncbi.nlm.nih.gov/pug_rest/pug_rest.xsd` |
| How often | at most 5 requests a second: one at a time, at least 0.2 s from one request's start to the next | Programmatic Access (May 19, 2025), its usage policy; the same at the head of PUG REST |
| How long | 30 s for an answer, "the default timeout on PubChem servers"; a whole lookup ends within 80 s, inside Meno's 90 s | Programmatic Access; PUG REST Tutorial, *Error Handling* |
| Being asked to go slower | the `X-Throttling-Control` header: where any of its statuses is Red or Black, the next request waits 1 s. The page gives no number: 1 s is the plugin's own choice, with no measured basis | Dynamic Request Throttling (August 19, 2026) |
| The statuses | 404 with the fault `PUGREST.NotFound`: no compound by this name. 400: "PubChem could not read this name". 503 ("Too many requests or server is busy, retry later", and how throttling refuses): asked once more after 2 s (the plugin's own wait), then "PubChem is busy; try again in a moment". 504, or no answer in time: "PubChem did not answer in time". Anything else: "PubChem answered what this plugin does not read (HTTP n)" | PUG REST, *Status Codes*; Dynamic Request Throttling |
| An HTML answer | a 503 may bring an HTML page: no answer not said to be JSON is read as JSON | PUG REST (the note on 503 at its head); Dynamic Request Throttling |
| The size of an answer | at most 1 MiB read; a larger one is refused | the plugin's own bound |

A connection that cannot be made - offline, or not allowed by Meno's
network gate - is "PubChem could not be reached". A redirect is not
followed. Nothing is kept on disk: the same name looked up twice goes out
twice.

## What it does not do

- English names as PubChem holds them, matched whole; no word search and
  no fuzzy search.
- Meno draws the structure PubChem gives for the name and names its id; it
  does not check it against any other source.
- It needs a Meno that knows the role *A structure found by its name*: an
  older Meno shows the plugin as for a newer Meno (the catalogue's two
  lists, README).
