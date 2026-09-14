# Övningsuppgifter

## Träning av ett neuralt nätverk för hand – Exempel 1
Det neurala nätverket nedan ska tränas till att kunna prediktera en hög utsignal Y vid udda antal höga insignaler X1 och X2, annars ska utsignal Y bli noll:

![](./images/network1.png)

Den första träningsuppsättningen: insignaler X1X2 = 01, Y = 1.

**Startparametrar:**

```
b1 = 0.2,  b2 = 0.4,  b3 = 0.6
w1 = 0.5,  w2 = 0.6,  w3 = 0.3
w4 = 0.8,  w5 = 0.1,  w6 = 0.9
```

Lärhastighet: LR = 0.1. Aktiveringsfunktion: ReLU för alla noder.

Genomför feedforward, backpropagation samt optimering. Beräkna sedan utsignalen igen. Blev felet mindre?

---

## Träning av ett neuralt nätverk för hand – Exempel 2
Betrakta det neurala nätverket nedan:

![](./images/network2.png)

Nätverket ska tränas till att prediktera utsignaler Y1Y2 som utgör inversen till insignaler X1X2:

| X1X2 | Y1Y2 |
|:----:|:----:|
|  00  |  11  |
|  01  |  10  |
|  10  |  01  |
|  11  |  00  |

Första träningsuppsättning: X1X2 = 10, Y1Y2 = 01.

**Startparametrar – dolda lagret:**

```
b1 = 0.1,  b2 = 0.1,  b3 = 0.7
w1 = 0.2,  w2 = 0.9,  w3 = 0.3
w4 = 0.8,  w5 = 0.5,  w6 = 0.4
```

**Startparametrar – utgångslagret:**

```
b4 = 0.1,  b5 = 0.6
w7  = 0.1,  w8  = 0.1,  w9  = 1.0
w10 = 0.3,  w11 = 0.0,  w12 = 0.6
```

Lärhastighet: LR = 0.1. Aktiveringsfunktion: ReLU för alla noder.

Genomför feedforward, backpropagation samt optimering. Beräkna sedan utsignalen igen. Blev felet mindre?

---

## Träning av ett neuralt nätverk för hand – Exempel 3
Betrakta det neurala nätverket nedan:

![](./images/network3.png)

Nätverket ska tränas till att kunna prediktera en hög utsignal Y vid udda antal höga insignaler X1 och X2, annars ska utsignal Y bli noll.

Träning har genomförts via en epok tidigare och parametrarna har vid start följande värden:

**Startparametrar:**

```
b1 = 0.1,   b2 = -0.4,  b3 = -0.2
w1 = 0.5,   w2 = 0.4,   w3 = -0.2
w4 = 0.1,   w5 = 0.7,   w6 = 0.8
```

Träning ska nu genomföras med träningsuppsättningen X1X2 = 11, Y = 0.

Lärhastighet: LR = 0.1. Aktiveringsfunktion: ReLU för alla noder.

Genomför feedforward, backpropagation samt optimering. Beräkna sedan utsignalen igen. Blev felet mindre? Om inte, vad hade du kunnat ändra för att erhålla mindre fel?

---
