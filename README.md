# Maskininlärning
Repo för kursen **Maskininlärning** med klassen Ee25, ht26 - vt27.

## Om kursen
Kursen behandlar grunderna inom maskininlärning med fokus på:
* Linjär regression samt gradientbaserad optimering.
* Neurala nätverk: feedforward, backpropagation samt aktiveringsfunktioner.
* Konvolutionella neurala nätverk (CNN) för bildklassificering: kernels, pooling- samt flatten-lager.
* Implementering av ML-algoritmer från grunden i C++, utan externa ML-bibliotek.
* Integrering av ML-modeller i inbyggda system (ESP32-S3).
* Ansvarsfull användning av AI-verktyg vid utveckling.

Under kursens gång implementeras:
* En adaptiv linjär regressionsmodell samt en smart temperatursensordriver.
* Ett enkelt neuralt nätverk (dense-lager) från grunden i C++.
* En valfri ML-algoritm för en egenvald tillämpning, integrerad i ett inbyggt system.
* Conv-, pooling- samt flatten-lager för konvolutionella neurala nätverk.

Efter genomförd kurs ska studenten kunna:
* Implementera grundläggande ML-algoritmer från grunden i C++.
* Träna samt utvärdera neurala nätverk, för hand och i mjukvara.
* Integrera ML-modeller i ett inbyggt system.
* Resonera kring designval, begränsningar samt lämplig användning av AI-verktyg.

---

## Struktur

```text
ci/          Skript för kodformattering och kompilering
exams/       Tentameninformation och övningstentamen
info/        Kursinformation, schema och examination
lectures/    Föreläsningar, litteratur och övningsuppgifter
libs/        Delade C++-bibliotek (t.ex. testramverket yrgo::test), som git-submoduler
projects/    Projektbeskrivningar, krav och inlämningsinstruktioner
```

---

## Kodexempel
[cnn_demo](./lectures/L13/cnn_demo) (L13) är en komplett C++17-implementation av ett konvolutionellt
neuralt nätverk som tränas att klassificera 4×4-pixelbilder av siffrorna 0-3. Demot visar hur lagren
hänger ihop från indata till prediktion:
* `conv_layer::Conv`: conv-lager med en kernel, en bias samt "same"-padding.
* `conv_layer::MaxPool`: icke-överlappande maxpooling.
* `flatten_layer::Flatten`: formar om 2D-matrisen till en 1D-vektor.
* `dense_layer::Dense`: fullt uppkopplat utgångslager med one-hot-kodad utdata.

Bygg och kör med `make` i katalogen; se [README](./lectures/L13/cnn_demo/README.md) för en
genomgång av arkitekturen.

---

## Kodformattering
`ci/format.sh` formaterar C/C++-kod med `clang-format` samt Python-kod med `black`:

```bash
ci/format.sh          # Formatera alla filer.
ci/format.sh --check  # Kontrollera formattering utan att ändra filer.
```

Samma sak går att köra via rotens `Makefile`:

```bash
make format        # Formatera alla filer.
make format-check  # Kontrollera formattering utan att ändra filer.
```

Innan skriptet körs behöver `clang-format` samt `black` finnas installerade och tillgängliga i
`PATH`.

`clang-format` installeras via `apt` i WSL:

```bash
sudo apt -y update
sudo apt -y install clang-format
```

`black` installeras via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---
