# L06 - Testsvit
Enhetstester för dense-lagerstubben från [bilaga B](../../appendix/b_exercises.md), skrivna med 
testramverket [`yrgo-test`](https://github.com/yrgo-libs/yrgo-test).

---

## Köra testerna
Kör följande kommando i denna katalog:

```bash
make
```

Om katalogen `libs/test` i repots rotkatalog är tom, hämta testramverket först:

```bash
git submodule update --init
```

Skriver ni er kod i er egen `ml`-kodbas i stället för i `exercises`, ange sökvägen till den:

```bash
make ML_DIR=<sökväg till er ml-katalog>
```

---
