# Bilaga B - Konvolutionella neurala nätverk (CNN)

### Översikt
Ett typiskt CNN består av fyra typer av lager:
1. **Konvolutionella lager** (hädanefter *conv-lager*) identifierar och extraherar lokala mönster
   och egenskaper i en bild.
2. **Pooling-lager** samplar ned bilden, vilket minskar dess storlek samtidigt som de viktigaste
   mönstren bevaras.
3. **Flatten-lager** plattar den tvådimensionella bilden till en endimensionell representation som
   kan matas in i vanliga neurala lager.
4. **Dense-lager** gör, baserat på de extraherade egenskaperna, den slutliga prediktionen.

De tre första lagren extraherar alltså attribut ur bilden, medan dense-lagret står för den
avslutande tolkningen.

---

### Konvolutionella lager (conv-lager)
Ett konvolutionellt lager applicerar en *kernel*, ett litet filter med vikter, över hela
inputbilden. Samma kernel återanvänds på varje position i bilden, vilket håller nere antalet
parametrar och gör att samma mönster kan kännas igen oavsett var i bilden det dyker upp. Varje
kernel letar efter ett specifikt lokalt mönster, till exempel en kant eller ett hörn; i praktiken
används ofta flera kernels per conv-lager för att fånga olika mönster samtidigt, men för att hålla
det enkelt använder vi bara en kernel per conv-lager i denna kurs.

Två egenskaper skiljer ett conv-lager från ett vanligt dense-lager:
* Beräkningarna sker lokalt, på ett litet fönster av bilden i taget.
* Samma vikter (kerneln) återanvänds över hela bilden i stället för att varje nod har sina egna. 

Utöver det fungerar det precis som ett vanligt lager, med viktad summa, bias och en
aktiveringsfunktion (vanligtvis ReLU). Lagrets output kallas en *feature map*, och själva processen att extrahera dessa 
mönster kallas *feature extraction*.

**Feedforward:** Kerneln glider över inputbilden ett steg i taget (oftast med `stride = 1`, det vill
säga ett pixelsteg åt gången). För varje position:
1. Ett litet fönster väljs ut ur bilden (t.ex. 2×2).
2. Värdena multipliceras med motsvarande kernel-vikter.
3. Resultatet summeras ihop med ett bias-värde.
4. Aktiveringsfunktionen appliceras på resultatet.

Värdet på varje position i outputen visar alltså hur starkt kerneln reagerade på just den platsen
i bilden.

**Backpropagation:** Precis som för dense-lager sprids felet bakåt genom lagret för att räkna fram
gradienter, det vill säga hur mycket och åt vilket håll varje vikt och bias-värde bör justeras.
Skillnaden är att eftersom samma kernel användes på alla positioner i bilden, måste gradienterna
från samtliga dessa positioner ackumuleras innan kernelns vikter uppdateras. Varje kernel-vikt har
med andra ord påverkats av många olika delar av bilden, och backpropagation räknar ut det
sammanlagda bidraget.

**Optimering:** Fungerar identiskt med dense-lager; varje kernel-vikt och bias-värdet justeras
längs sin gradient, skalat med lärhastigheten. Gradienten adderas i stället för att subtraheras,
eftersom avvikelsen beräknas som `y_ref - y_p` och därmed redan pekar i den riktning som minskar
felet. Skillnaden är bara att det handlar om
betydligt färre parametrar, eftersom ett conv-lager bara består av en liten kernel och ett
bias-värde snarare än en unik vikt per nod och input.

---

### Pooling-lager
Pooling-lager samplar ned bilden så att nätverket får mindre data att bearbeta, vilket både
förenklar träning/prediktion och gör nätverket mer robust mot små variationer i indatan.
Pooling-lager saknar helt träningsbara parametrar. Indatan delas in i block (t.ex. 2×2), och ur
varje block plockas ett enda värde ut:
* **Max pooling:** det största värdet i blocket. Klart vanligast, och det vi använder i den här
  kursen.
* **Average pooling:** genomsnittet av blocket.

**Feedforward:** Det största värdet i varje pooling-block skickas vidare. De mest framträdande
attributen i varje region bevaras, på bekostnad av detaljer, ungefär som när en bild samplas ned:
den ser i stort sett likadan ut, men förstorar man den igen syns det att detaljer gått förlorade.

**Backpropagation:** Gradienterna skickas bara tillbaka till de positioner som faktiskt hade
maxvärdet vid feedforward; övriga positioner får gradienten 0, eftersom de inte bidrog till
outputen.

---

### Flatten-lager
Efter konvolution och pooling är datan fortfarande tvådimensionell, medan dense-lager kräver
endimensionell indata. Flatten-lagret löser det genom att platta om datan från 2D till 1D. Det
utför inga beräkningar och har inga egna vikter; det ändrar bara datans form. Vid backpropagation
gör det motsvarande operation baklänges, och formar om gradienten från 1D tillbaka till 2D.

---

### Dense-lager i CNN
Dense-lagret fungerar exakt som i ett vanligt neuralt nätverk. Den enda skillnaden är att indatan nu
redan består av extraherade, komprimerade features snarare än rådata, och lagret väger samman dessa
features för att göra den slutliga klassificeringen eller regressionen.

---

### CNN i ett nötskal
Ett CNN kan ses som ett vanligt neuralt nätverk med några extra förbehandlingslager specialiserade på rumslig data:
* Conv-lagret letar efter lokala mönster.
* Pooling-lagret behåller de mest framträdande av dem och slänger resten.
* Flatten-lagret gör om datan till en form dense-lagret förstår (en dimension).
* Dense-lagret gör till sist den faktiska tolkningen av informationen. 

Backpropagation följer samma kedja baklänges:
* Dense-lagret justerar hur features viktas.
* Flatten-lagret formar om gradienterna.
* Pooling-lagret skickar gradienterna till rätt positioner.
* Conv-lagret justerar till slut vikterna för de mönster som faktiskt bidrog till prediktionen.

---
