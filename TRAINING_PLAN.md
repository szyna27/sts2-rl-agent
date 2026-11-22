# Plan treningu Combat AI i Metagame

Ten dokument opisuje docelowy przepływ uczenia agenta: najpierw polityka walki,
a następnie wybory poza walką w pełnym runie. Oddziela założenia od tego, co
jest już zaimplementowane. Samo istnienie kodu treningowego nie oznacza, że
model został już wytrenowany ani zweryfikowany na docelowej liczbie runów.

## Podsumowanie

| Obszar | Cel | Stan |
|---|---|---|
| Combat AI | Uczyć politykę walki na zróżnicowanych stanach i przeciwnikach | Generator talii oraz curriculum Aktów 1/4 → 2 → 3 są zaimplementowane; trening PPO należy uruchomić i ocenić |
| Metagame | Uczyć wyboru mapy i decyzji runowych przy użyciu wytrenowanej, zamrożonej polityki walki | Trening pełnych runów i integracja modelu są zaimplementowane; pozostaje trening właściwym modelem combat i ewaluacja |
| Pełny zakres gry | Uwzględnić dalszy endgame | Symulator pełnego runu obsługuje obecnie trzy akty i nie modeluje walki z Heart |

## 1. Architektura i kolejność uczenia

Agent składa się z dwóch polityk:

1. **Combat AI** wybiera legalne akcje w walce. Jest trenowana jako
   `MaskablePPO` z maską legalnych akcji.
2. **Metagame AI** wybiera działania poza walką: trasę na mapie, karty i
   nagrody, relikty, opcje sklepu, odpoczynku oraz eventów. Ocenia symulowane
   stany po każdej dostępnej decyzji.

Docelowa kolejność:

1. Wytrenować i ocenić model Combat AI na curriculum opisanym niżej.
2. Zamrozić wybrany model Combat AI i użyć go do rozgrywania walk podczas
   treningu Metagame AI.
3. Porównać Metagame AI z prostą polityką heurystyczną na tych samych seedach.
4. Dopiero po stabilnej ewaluacji połączyć oba modele w bridge do gry.

Metagame nie powinno trenować ani zmieniać polityki combat w trakcie własnych
epizodów. Dzięki temu nagroda za decyzje runowe nie miesza się z uczeniem
wykonywania tur walki.

## 2. Trening Combat AI

### 2.1 Zasady ogólne

- Curriculum obejmuje akty 1–3 oraz alternatywny Akt 4.
- **Akt 1 i Akt 4 stanowią jeden wspólny etap.** Pierwszy z nich jest wybierany
  losowo, a akty są przełączane po każdym epizodzie. Żaden nie jest trenowany
  dopiero po drugim.
- Po zaliczeniu etapu 1/4 odblokowywane są kolejno Akt 2 i Akt 3.
- Trening ma prezentować różne rozmiary talii, ulepszenia, klątwy i typy starć,
  zamiast uczyć model tylko jednej, statycznej konfiguracji.
- Podczas treningu zbierane są statystyki per karta: średnia utrata HP w
  walkach, w których karta była w talii, oraz osobno w walkach, w których
  została zagrana. Wyniki są rozdzielone per encounter i agregowane per akt /
  typ starcia (np. wspólny licznik dla wszystkich Act 1 Weak).
- W pierwszym etapie treningu mikstury nie są celem optymalizacji.
- Domyślny punkt startowy walki to pełne HP. Utracone HP jest kosztem w nagrodzie.
- Ewaluacja curriculum używa stałego zestawu seedów, a przejście dalej jest
  uzależnione od win-rate'u.

Notebook [combat_rl_training.ipynb](combat_rl_training.ipynb) ma obecnie etapy
`(1, 4) → 2 → 3`, progi odpowiednio `65%`, `55%` i `45%` oraz `100 000`
dodatkowych kroków na etap w jednym uruchomieniu. Domyślne `TRAIN_MODE = 'load'`
pomija trening; należy ustawić `'new'`, aby rozpocząć od zera, lub `'continue'`,
aby wznowić zapisany etap.

### 2.2 Generator talii: Akt 1 i alternatywny Akt 4

Usuwanie kart startowych jest modelowane jako zdarzenia rozłączne. Usuwane są
wyłącznie bazowe karty Strike lub Defend.

| Starcie | Usunięcie 1 / 2 kart | Common | Uncommon | Rare | Ulepszenia | Greed |
|---|---:|---:|---:|---:|---:|---:|
| Weak | 15% / 10% | 0–3 | 1 (25%) | 1 (5%) | 5% na kartę, maks. 1 | 10% |
| Normal | 25% / 15% | 2–5 | 1–2 | 1 (15%) | 10% na kartę, maks. 2 | 10% |
| Elite | 25% / 15% | 0–5 | 0–2 | 1 (10%) | 10% na kartę, maks. 2 | 10% |
| Boss | 50% / 25% | 4–6 | 1–3 | 1 (50%) | 15% na kartę, maks. 3 | 10% |

| Starcie | Relikt startowy Neow | Common | Uncommon / Rare | Sklep |
|---|---:|---:|---:|---:|
| Weak | 1 (100%) | 1 (15%) | 1 (5%) | 0% |
| Normal | 1 (100%) | 0–1 (50%) | 0–1 (25%) | 1 (10%) |
| Elite | 1 (100%) | 0–1 (40%) | 0–1 (20%) | 1 (10%) |
| Boss | 1 (100%) | 1–3 | 1 (60%) | 1 (25%) |

Greed jest jedyną klątwą w tym profilu.

### 2.3 Generator talii: Akty 2 i 3

Przed walką losowany jest archetyp rozmiaru talii: mała (30%), średnia (45%)
lub duża (25%). Archetyp ogranicza ryzyko, że polityka nauczy się grać wyłącznie
jednym rozmiarem talii.

| Akt / archetyp | Usunięcie Strike/Defend | Common | Uncommon | Rare | Ulepszenia | Losowe klątwy |
|---|---:|---:|---:|---:|---:|---:|
| Akt 2 — mała (wszystkie walki) | 3–5 | 2–4 | 2–3 | 1–2 | 35% na kartę | 1 (15%) |
| Akt 2 — średnia (Normal/Elite) | 1–3 | 5–8 | 3–5 | 1–3 | 25% na kartę | 1 (25%) |
| Akt 2 — duża (Normal/Elite) | 0–1 | 8–12 | 5–8 | 2–4 | 20% na kartę | 1–2 (35%) |
| Akt 2 — Boss | 2–4 (50%) | +1 do puli | +1 do puli | +1 (70%) | +10 punktów proc. | bez zmian |
| Akt 3 — mała (wszystkie walki) | 5–8 | 2–5 | 2–4 | 2–3 | 50% na kartę | 1 (20%) |
| Akt 3 — średnia (Normal/Elite) | 2–4 | 6–10 | 4–7 | 2–4 | 40% na kartę | 1–2 (30%) |
| Akt 3 — duża (Normal/Elite) | 0–2 | 10–16 | 6–10 | 3–6 | 30% na kartę | 1–2 (45%) |
| Akt 3 — Boss | bez zmian | +1–2 do puli | +1–2 do puli | +1 (80%) | +10 punktów proc. | bez zmian |

Generator ma zachować co najmniej 2–3 karty zadające obrażenia lub pozwalające
skalować obrażenia, nawet po usunięciu kart startowych.

### 2.4 Relikty i HP

Docelowa liczba reliktów ma rosnąć wraz z aktem i typem starcia:

| Etap | Startowy | Relikty bossów | Common | Uncommon | Rare | Sklep |
|---|---:|---:|---:|---:|---:|---:|
| Akt 1/4 — Weak | 1 | — | 1 (15%) | — | 1 (5%) | 0% |
| Akt 1/4 — Normal | 1 | — | 0–1 (50%) | 0–1 (25%) | — | 1 (10%) |
| Akt 1/4 — Elite | 1 | — | 0–1 (40%) | 0–1 (20%) | — | 1 (10%) |
| Akt 1/4 — Boss | 1 | — | 1–3 | 1 (60%) | — | 1 (25%) |
| Akt 2 — Normal/Elite | 1 | 1 z Aktu 1 | 2–4 | 1–2 | 0–1 (50%) | 0–1 (40%) |
| Akt 2 — Boss | 1 | 1 z Aktu 1 | 3–5 | 2–3 | 1 (75%) | 1 (50%) |
| Akt 3 — Normal/Elite | 1 | po 1 z Aktów 1 i 2 | 4–6 | 2–4 | 1–2 | 1–2 (60%) |
| Akt 3 — Boss | 1 | po 1 z Aktów 1 i 2 | 5–7 | 3–5 | 2–3 | 1–2 (75%) |

Relikty bossów z ograniczeniami muszą prawidłowo wpływać na symulację (np.
Runic Dome ukrywa intencje wrogów). Liczniki, takie jak Pen Nib, Nunchaku,
Happy Flower i Incense Burner, powinny zaczynać walkę z losowym ładunkiem od
zera do `max_charge - 1`. Relikty działające wyłącznie poza walką nie powinny
zaśmiecać stanu wejściowego.

W Aktach 2 i 3 docelowo 20% walk zaczyna się przy 40–65% maksymalnego HP.
Pozostałe walki zaczynają się przy pełnym HP. Ta zmienność ma uczyć obsługi
efektów zależnych od niskiego zdrowia i trudniejszych decyzji obronnych.

### 2.5 Stan implementacji Combat

- [x] Generator talii dla Aktów 1–4, archetypy Aktów 2–3 i minimalna liczba kart
  ofensywnych.
- [x] Środowisko treningowe może losować akty, a etap `[1, 4]` wybiera losowy
  początek i przełącza akt po każdym resecie.
- [x] Notebook curriculum, progi przejścia, ewaluacja i wznowienie checkpointu.
- [x] Telemetria kart zapisywana okresowo do `card_performance.csv` dla
  encounterów i agregatów akt/typ.
- [ ] Generator profili reliktów, relikty bossów i losowe liczniki.
- [ ] Losowanie obniżonego HP dla części walk Aktów 2–3.
- [ ] Uruchomienie właściwego treningu PPO i ocena win-rate'u na seedach
  ewaluacyjnych dla każdego etapu.

## 3. Trening Metagame

### 3.1 Cel i model decyzji

Metagame ma maksymalizować wynik całego runu, a nie pojedynczej walki.
`MetagameAgent` rozważa legalne akcje `RunManager`, symuluje każdą z nich na
kopii stanu i koduje uzyskany stan. Sieć wartości ocenia kandydatów, a agent
wybiera najlepiej oceniany. Symulacja kandydatów nie powinna modyfikować
aktywnego runu.

Kod stanu obejmuje m.in. talię i relikty, HP i maksymalne HP, złoto, piętro,
akt i piętro aktu, poziom ascension, dostępne bossy, fazę runu, typ pokoju oraz
cechy kandydującej akcji. Ten sam schemat jest używany do wyborów w bridge.
Schemat zawiera cechy kluczy do Aktu 4, ale bieżący enkoder `RunManager`
wpisuje dla nich zero; należy je podłączyć, gdy symulator będzie modelować ten
fragment runu.

### 3.2 Dane treningowe i nagroda

- Każdy epizod rozgrywa pełny run zarządzany przez `RunManager`.
- Model combat jest **zamrożony**. Może to być model `MaskablePPO` albo
  heurystyka użyta jako baseline do wstępnych testów.
- Metagame wybiera wyłącznie akcje poza walką. W walce uruchamiana jest
  przekazana polityka combat.
- Trening używa eksploracji epsilon-greedy, replay buffera, sieci targetowej
  oraz bootstrapowanego celu wartości stanu.
- Aktualna nagroda wynosi: `+1` za każde przejście piętra, `+10` za wygraną
  walkę z elite, `+50` za wygraną walkę z bossem, `+100` za wygraną runu oraz
  `-50` za przegraną runu.
- Domyślne parametry treningu to 1000 epizodów, seed `20261112`, `gamma=0.99`,
  learning rate `1e-3`, batch size 32, replay buffer 20 000, epsilon
  `1.0 → 0.05` przez 10 000 decyzji, synchronizacja target network co 100
  aktualizacji i warstwy ukryte o rozmiarze 256.

Punkt wejścia to `python scripts/train_metagame.py`. Należy podać dokładnie
jeden z argumentów `--combat-model` lub `--combat-heuristic`; wynikowy
checkpoint zapisuje schemat cech i metadane treningu.

### 3.3 Integracja i ograniczenia

- [x] Sieć wartości, enkoder stanu i agent wyboru decyzji.
- [x] Trening pełnych runów z zamrożoną polityką combat, reward shaping,
  zapisem checkpointu i logowaniem postępu.
- [x] Bridge może użyć modelu metagame do wyborów mapy, kart/nagród, sklepu,
  odpoczynku, eventów, skarbów i reliktów bossów.
- [ ] Wytrenować i ocenić model metagame z właściwym, wytrenowanym modelem
  Combat AI zamiast polegać wyłącznie na heurystyce.
- [ ] Raportować na wspólnych seedach win-rate runów, średnie piętro, odsetek
  dojścia do bossów, zwroty i porównanie z heurystykami.
- [ ] Rozszerzyć pełny run o docelowy endgame, w tym klucze/Heart. Aktualny
  symulator `RunManager` obejmuje trzy akty i nie symuluje walki z Heart;
  Akt 4 używany w curriculum combat jest alternatywną pulą walk, a nie
  dodatkowym aktem pełnego runu.

## 4. Plan prac od teraz

1. Dokończyć realistyczne stany combat: relikty i ich liczniki, relikty
   bossów, HP dla późnych aktów oraz testy ich efektów w obserwacji.
2. Uruchomić notebook od etapu `[1, 4]`; zbierać osobne checkpointy i
   ewaluacje dla kolejnych etapów. Nie rozpoczynać Aktu 2 przed zaliczeniem
   wspólnego etapu Aktów 1/4.
3. Po uzyskaniu stabilnej polityki combat uruchomić trening metagame z tym
   checkpointem jako zamrożoną polityką walki.
4. Porównać metagame i heurystyki na tych samych seedach. Oceniać zarówno
   wynik całego runu, jak i postęp na mapie; nie opierać decyzji wyłącznie na
   średniej nagrodzie.
5. Po walidacji obu modeli wykonać test end-to-end przez bridge, a następnie
   zdecydować o rozszerzeniu symulatora o dalszy endgame.

Nie należy uznawać etapu za ukończony tylko dlatego, że kod treningowy
istnieje: wymagane są zapisane wyniki ewaluacji i checkpoint modelu.
