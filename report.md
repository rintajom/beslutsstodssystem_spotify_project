# Rapport - Anette & Minervas rekommendationssystem av Spotify sånger

## EDA

Vi började med att leta fram ett dataset som intresserade oss. Vi har gjort några projekt med film-data så vi tänkte att musik-data sku vara något nytt. Originellt ville vi använda detta dataset som var rekommenderat https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered, men det visade sig att det inte går att ladda ner mera så vi hittade en likande från Kaggle. Datasetet vi hittade var denna: https://www.kaggle.com/datasets/solomonameh/spotify-music-dataset. 

Det första vi gjorde var att utforska kolumnerna i datasetet (/data/spotify_data.csv), men insåg ganska snabbt att det fanns lite väl många kolumner. Vi bestämde att behålla allt med sång namnet och artisten, men att utsluta bland annat spellist data. Det fanns flera kolumner om sång kvalitet såsom "liveness", "valence" och "acousticness" som vi inte ansåg var nödvändiga. Däremot valde vi att hålla "tempo" och "loudness" eftersom det hade värden (bpm och decibel) som kan mätas. Dessutom valde vi "danceability" för att vi tyckte en rolig kolumn att hålla. 

I efterhand insåg vi att vi var kanske lite väl stränga med att droppa kolumner. Vi tog bort allt som hade med spellistor att göra, vilket ledde till att också t.ex. "playlist_genre" föll bort. Det skulle vi ha haft nytta av, men vi valde att gå in en annan riktning på grund av våra tidigare val. 


## Data cleanup

Vi började med att städa upp data i filen data_cleanup.py. Vi tog bort flera kolumner som vi inte upplevde att behövdes för projektet. Vi skapade också en egen id kolumn för datat eftersom det som fanns var lite för komplext. Vi raderade duplikater och tomma värden. Vi scalea de numeriska värdena och begränsade dem till två decimaler. Detta sparade vi sedan till en ny csv (/data/cleaned_spotify_data.csv).


## Kodandet

Vi följde ganska mycket tidigare uppgifter vi haft under i kursen, men vi måste justera så att det fungerade med vårat dataset. 

**main.py**
Filen kör alla rekommendationssytem, med sången som man väljer (test_song). Filen hämtar den rensade datan och initierar de tre rekommendationssystemen (content-based, rule-based och hybrid). Filen printar den valda sångens info och generar sedan rekommendationer med alla metoder. Sist utvärderas rekommendationerna som printtas i en tabell. 

**contentBasedRecommender.py**
I filen körs content-based rekommendationssystem som använder ML för att hitta sånger som liknar varandra. Systemet analyserar både texten i sångnamnen (med TF-IDF-vectorization) och audio egenskaper (tempo, loudness och danceability). Likheten räknas ut genom att addera cosine similarity för texten och euclidean distance för audio.

**ruleBasedRecommender.py**
Vi skulle först använda collaborative-filtering men eftersom vårt dataset inte har ratings så blev det omöjligt. Därför valde vi rule-based som fungerade ba med våra numeriska kolumner.

I filen körs alltså en rule-based rekommendationssystem som använder matematiska regler för att beräkna hur lika sånger är. Det fokuserar på audio (tempo, loudness, danceability) och popularitet och räknar ut hur mycket varje sång skiljer sig från test_song. Varje feature vägs samman till en similarity score, där populariteten får lite extra vikt (multiplicerat med 0.5). Resultaten normaliseras mellan 0 och 1 med MinMaxScaler, så att de går att jämföra.

**hybridRecommender.py**
Filen kombinerar resultaten från både content-based och rule-based rekommendationssystemen för att skapa en hybridrekommendation. Den tar rekommendationer från båda systemen och kombinerar dem. Den itererar genom båda listorna och väljer fem sånger. Såhär ser vi till att både content-based och rule-based rekommendationer visas.

**evaluationVerification.py** 
Filen har tre funktioner för att evaluera och verifiera rekommendationerna. 
- mean_feature_distance: mäter hur långt från test_song de rekommenderade sångerna är, baserat på audio egenskaper
- mean_absolute_error: mäter hur stort det genomsnittliga felet är mellan audio egenskaper
- mean_feature_correlation: mäter hur väl audio egenskaper mellan test_song och rekommendationerna hänger ihop


## Resultat och analys

Vi valde att testa rekommendationerna med sången "Good luck, Babe!" aka test_song.


| Track info: |  |
|---------------|-----------|
| Track | Good Luck, Babe! |
| Artist | Chappell Roan |
| Album | Good Luck, Babe! (2024-04-05) |
| Popularity | 0.81 |
| Tempo*| 0.42 BPM |
| Loudness | 0.84 dB |
| Danceability | 0.67 |

---------
**Content-Based Recommendations:**

|          | **Track** | **Artist** | **Popularity** | **Similarity** |
|----------|-----------|------------|----------------|----------------|
| 1 | Good To Be | Mark Ambor | 0.09 | 0.69 |
| 2 | Good Love | Hannah Laing, RoRo | 0.09 | 0.64 |
| 3 | Good Feeling | Flo Rida | 0.25 | 0.62 |
| 4 | Hurts So Good | John Mellencamp | 0.03 | 0.61 |
| 5 | Good Time | Owl City, Carly Rae Jepsen | 0.22 | 0.61 |

Här kan vi se att den rekommenderar enligt sångnamn. Detta lyckas den bra med, vi ser att alla sånger har ordet "Good" i sig. Den högst rankade sången har en similarity score på 69% vilket inte är så högt, men detta är ganska bra med tanke på att sångens namn måst vara liknande som test_song namnet. 


-------
**Rule-Based Recommendations:**

|          | **Track** | **Artist** | **Popularity** | **Similarity** |
|----------|-----------|------------|----------------|----------------|
| 1 | Tu Boda | Oscar Maydon, Fuerza Regida | 0.78 | 0.96 |
| 2 | Who | Jimin | 0.75 | 0.94 |
| 3 | Timeless (with Playboi Carti) | The Weeknd, Playboi Carti | 0.75 | 0.93 |
| 4 | Espresso | Sabrina Carpenter | 0.69 | 0.93 |
| 5 | EL LOKERON | Tito Double P | 0.66 | 0.93 |

Här ser vi också ett ganska bra resultat. Den kollar här efter audio egenskaperna. Då vi lyssnade på sångerna hade de stora likheter med test_song, speciellt med bland annat tempo. Det var roligt att se att sångerna ända var väldit olika, t.ex. olika språk och stiler. 

----------
**Hybrid Recommendations:**

|          | **Track** | **Artist** | **Popularity** | **Similarity** |
|----------|-----------|------------|----------------|----------------|
| 1 | Good To Be | Mark Ambor | 0.09 | 0.69 |
| 2 | Tu Boda | Oscar Maydon, Fuerza Regida | 0.78 | 0.96 |
| 3 | Good Love | Hannah Laing, RoRo | 0.09 | 0.64 |
| 4 | Who | Jimin | 0.75 | 0.94 |
| 5 | Good Feeling | Flo Rida | 0.25 | 0.62 |

Här väljer den en bra blandning av de två tidigare rekommendationerna. En blandning ser till att flera faktorer beaktas i rekommendationen, vilket kan ge ett bättre resultat.

--------
**Evaluation Metrics:**

| **Metric** | **Content-Based** | **Rule-Based** | **Hybrid** |
|------------|-------------------|----------------|------------|
| Mean Feature Distance | 0.1397 | 0.0637 | 0.1019 |
| Mean Absolute Error | 0.0660 | 0.0293 | 0.0480 |
| Mean Feature Correlation | 0.9601 | 0.9911 | 0.9944 |

- MFD: Ju högre värde desto större är distansen från test_songs audio features (tempo, loudness och danceability). Alltså ger content-based rekommendationer som är mest olika från test_song, medan rule_base ger en väldigt bra rekommendation med tanke på audio features.

- MAE: Här ser vi även att rule-based rekommendationen får minst andra värden av audio egenskaper jämförelse med test_song. Detta tyder på att rule-based rekommendationerna är mest lik test_songs audio egenskaper.

- MFC: Hybrid rekommendationerna vinner här. Vilket betyder att den är bäst på att hålla balans mellan olika audio features. Det var intressant att hybrid fick bäst resultat, men det beror troligen på att den väljer de bästa sångerna av de två andra rekommendationerna.


## Övriga kommentarer

Vi har jobbat tillsammans under hela projektet. Alltså har det ingen skillnad vem som har gjort mera commits. Vi brukar arbeta väldigt bra tillsammans och tycker om att träffas i skolan och jobba sida vid sida. Detta hjälper oss att inte fastna upp och hitta på flera lösningar. Vi har även tagit hjälp av AI-verktyg, främst för att lösa errors och formattering.

Projektet var roligt och inspirerande. Vi tänker på att kanske utöka detta till att rekommendera sånger till oss baserat på våra egna spotify listor. 