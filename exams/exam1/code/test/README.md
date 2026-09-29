# Tester
Unit- och komponenttester för `ml`-kodbasen, skrivna med testramverket
`yrgo::test` (se [libs/test](../../../../libs/test/README.md)).

Innan testerna körs, se till att git-submodulerna är initierade och uppdaterade:

```bash
git submodule update --init --recursive
```

Bygg och kör därefter testerna med:

```bash
make
```

---

## Exempeltester
Katalogen innehåller sex exempeltester som samtliga går igenom: tre för `Dense` och tre för
`Shallow`. De visar mönstren ni bygger vidare på, bland annat hur ett förväntat värde räknas fram
ur lagrets egna vikter och hur `Stub::setOutput()` används, och de avslöjar ingen av de planterade
buggarna.

Felmeddelanden som `Feedforward failed due to dimension mismatch` skrivs ut av de testfall som
medvetet anropar med felaktiga argument. De betyder alltså inte att ett test har misslyckats.

---

