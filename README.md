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

## Kodformattering
`ci/format.sh` formaterar C/C++-kod med `clang-format` samt Python-kod med `black`:

```bash
ci/format.sh          # Formatera alla filer.
ci/format.sh --check  # Kontrollera formattering utan att ändra filer.
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
