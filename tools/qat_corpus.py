"""QAT corpus v2: broader coverage than v1 — all 33 registry voices, 22
texts across the 7 Ephraim languages incl. three long multilingual
passages, sampled combos. Appends to the same dump file (S2P_DUMP_FRAMES
opens "ab"), so the v1 frames stay valid as a prefix.

Deliberately EXCLUDES the two demo texts used for the final listening
takes (short Sprachentag, LONG_DE_EN) so the deliverables are not
generated from training material. (The v1 corpus contained the short
Sprachentag; documented in docs/QUANT.md.)"""
import json, random, time, urllib.request

MEDIUM = [
    # --- v1 texts (minus the Sprachentag demo text) ---
    ("Der Unterricht beginnt heute später, weil die Lehrkräfte eine Fortbildung besuchen. "
     "Bitte nutzt die Zeit, um eure Referate vorzubereiten. In der Bibliothek ist Platz reserviert. "
     "Wer Fragen zur Projektwoche hat, kommt in der großen Pause ins Sekretariat. "
     "Die Anmeldelisten hängen ab morgen am schwarzen Brett."),
    ("Science club meets on Thursday afternoon in the chemistry lab. "
     "We will build small water rockets and measure how high they fly. "
     "Bring an empty plastic bottle if you have one at home. "
     "Safety goggles are provided, and parents are welcome to watch the launches on the sports field."),
    ("Aujourd'hui, la cantine propose un menu spécial: gratin de légumes, salade verte et tarte aux pommes. "
     "Bon appétit à toutes et à tous! N'oubliez pas de rapporter vos plateaux."),
    ("El próximo viernes celebramos el día del libro. Cada clase presentará su historia favorita. "
     "Podéis traer libros de casa para el mercadillo de intercambio. ¡Será un día maravilloso!"),
    ("Сегодня после уроков состоится репетиция школьного хора. Приходите в актовый зал к трём часам. "
     "Мы готовим песни к весеннему концерту, и каждый голос важен."),
    ("Завтра о десятій годині відбудеться шкільна олімпіада з математики. "
     "Не хвилюйтеся: головне — спробувати свої сили. Бажаємо всім успіху!"),
    ("Yarın okul bahçesinde bahar şenliği var. Müzik, oyunlar ve yiyecek stantları olacak. "
     "Ailelerinizi de davet edebilirsiniz. Hep birlikte güzel bir gün geçireceğiz."),
    ("Die Theater-AG sucht noch Mitspielerinnen und Mitspieler für das Sommerstück. "
     "Auditions are on Wednesday — no experience needed, just bring your energy! "
     "Wir proben zweimal pro Woche, et la première aura lieu en juillet. "
     "¡Anímate y participa! Це буде незабутньо."),
    ("In der nächsten Woche stehen die Klassenfahrten an. Klasse sieben fährt an die Nordsee, "
     "Klasse acht in den Harz. Denkt an wetterfeste Kleidung, gute Schuhe und euren Ausweis. "
     "Die Busse fahren pünktlich um acht Uhr vom Parkplatz hinter der Turnhalle ab. "
     "Wer Medikamente braucht, gibt sie bitte vorher bei der Klassenleitung ab. "
     "Wir wünschen allen eine wunderbare Fahrt mit vielen neuen Eindrücken!"),
    # --- v2 additions ---
    ("Achtung, eine kurze Durchsage: Der Wandertag am Freitag findet bei jedem Wetter statt. "
     "Treffpunkt ist um Viertel nach acht am Haupteingang. Denkt an Regenjacken, festes Schuhwerk "
     "und genug zu trinken. Die Route führt über den Aussichtsturm zum Waldspielplatz und zurück."),
    ("The school library has new opening hours: Monday to Thursday from eight to four, "
     "and Friday until noon. Our spring reading challenge starts next week — read five books, "
     "collect five stamps, and win a surprise. Ask Ms. Carter at the front desk for your reading pass."),
    ("La semaine prochaine, la classe de sixième visitera le musée des sciences naturelles. "
     "Le départ est prévu à neuf heures devant le collège. Pensez à apporter un pique-nique "
     "et un carnet pour prendre des notes. L'exposition sur les volcans est vraiment impressionnante."),
    ("El taller de teatro comienza el martes en el aula de música. Ensayaremos una obra corta "
     "sobre un viaje fantástico. No hace falta experiencia, solo ganas de jugar y de imaginar. "
     "Las inscripciones están abiertas hasta el lunes en la secretaría."),
    ("Школьная газета ищет юных журналистов! Если ты любишь писать, фотографировать или рисовать, "
     "приходи в среду в кабинет литературы. Первый выпуск выйдет в конце месяца, "
     "и в нём будет репортаж о спортивном празднике."),
    ("У четвер наш клас поїде на екскурсію до ботанічного саду. Візьміть із собою зошити, "
     "олівці та гарний настрій. Ми побачимо рідкісні рослини з усього світу "
     "і дізнаємося, як працює оранжерея взимку."),
    ("Satranç kulübü her çarşamba öğle arasında kütüphanede toplanıyor. "
     "Yeni başlayanlar için küçük bir kurs da var. Ay sonunda okul turnuvası yapılacak "
     "ve kazananlar ilçe yarışmasına katılacak. Herkesi bekliyoruz!"),
    ("Liebe Eltern, dear parents: Der Elternabend der siebten Klassen findet am Donnerstag um "
     "neunzehn Uhr statt. We will talk about the upcoming exchange programme and the new timetable. "
     "Eine Übersetzung ins Englische wird angeboten. Wir freuen uns auf Ihr Kommen!"),
    ("Bitte notiert euch diese Termine: Am 14. März um 8:45 Uhr beginnt die Matheprüfung, "
     "am 21. März um 10:30 Uhr die Präsentation der Projektwoche, und am 3. April fahren wir "
     "um 7:15 Uhr zum Schwimmwettkampf. Die 42 Teilnehmerplätze werden bis zum 28. Februar vergeben."),
]

LONG = [
    ("Liebe Schülerinnen und Schüler, liebe Kolleginnen und Kollegen, herzlich willkommen zu unserem "
     "großen Schulfest! Seit Wochen haben alle Klassen gebastelt, geprobt und geplant — heute zeigen wir, "
     "was in unserer Schule steckt. Auf dem Hof findet ihr Spielstationen, eine Tombola und die berühmte "
     "Waffelbude der Klasse acht. In der Aula treten ab elf Uhr die Musikklassen auf, danach folgt das "
     "Improvisationstheater. Dear guests, a very warm welcome to our school festival! All morning you can "
     "explore project stands, watch performances, and taste dishes from many countries. Our students have "
     "worked hard, and they are excited to show you their ideas. Chers parents, soyez les bienvenus! "
     "N'hésitez pas à poser des questions aux élèves — ils adorent expliquer leurs projets. Le spectacle "
     "de danse commence à quatorze heures dans le gymnase. Und noch ein Hinweis: Der Erlös des Festes geht "
     "in diesem Jahr an die neue Leseecke der Bibliothek. Wer beim Aufräumen helfen möchte, meldet sich "
     "bitte am Infostand. Und jetzt: Lasst uns gemeinsam feiern — viel Spaß euch allen!"),
    ("Liebe Absolventinnen und Absolventen, heute ist euer Tag! Vor vielen Jahren seid ihr mit großen "
     "Schulranzen und noch größeren Augen hier angekommen — heute steht ihr selbstbewusst auf dieser Bühne. "
     "Ihr habt Formeln gelernt und wieder vergessen, Referate überlebt und Freundschaften geschlossen, die "
     "bleiben werden. Queridas familias, hoy celebramos juntos un gran logro. Estos jóvenes han crecido, "
     "han superado exámenes difíciles y han aprendido a no rendirse. Estamos muy orgullosos de cada uno de "
     "ellos. El futuro les espera con las puertas abiertas. Sevgili mezunlar, bugün sizin gününüz! "
     "Yıllarca birlikte çalıştık, güldük, bazen de zorlandık. Ama her zorluk sizi daha güçlü yaptı. "
     "Bundan sonra hangi yolu seçerseniz seçin, bu okulun kapısı size her zaman açık. Und damit, liebe "
     "Abschlussklasse: Nehmt euren Mut mit, eure Neugier und euren Humor. Macht Fehler, lernt daraus, und "
     "bleibt einander verbunden. Herzlichen Glückwunsch — und nun werft die Hüte in die Luft!"),
    ("Guten Morgen und herzlich willkommen an alle neuen Fünftklässlerinnen und Fünftklässler! Heute "
     "beginnt für euch ein neues Kapitel, und die ganze Schule freut sich auf euch. Eure Klassenlehrerinnen "
     "zeigen euch gleich die Gebäude, den Musikraum und die Turnhalle. Good morning, everyone! If you are "
     "ever unsure where to go, just ask an older student — helping each other is our most important rule "
     "here. Bonjour à toutes et à tous! Dans notre école, on apprend aussi les langues en chantant, en "
     "jouant et en cuisinant. Buenos días: cada año hacemos un intercambio con nuestra escuela amiga en "
     "Valencia, y quizá pronto participéis vosotros. Доброе утро! В нашей школе есть кружки по шахматам, "
     "робототехнике и рисованию — выбирайте, что вам нравится. Доброго ранку! Наша шкільна бібліотека "
     "відкрита щодня, і там завжди можна знайти тихе місце для читання. Günaydın! Okulumuzda herkes "
     "kendini evinde hissetsin istiyoruz. Und jetzt wünsche ich euch allen einen wunderbaren ersten "
     "Schultag — auf geht's in euer neues Abenteuer!"),
]

VOICES = ["achernar", "achird", "algenib", "algieba", "alnilam", "aoede",
          "autonoe", "callirrhoe", "charon", "deeper-male", "despina",
          "enceladus", "erinome", "fenrir", "gacrux", "iapetus", "kore",
          "laomedeia", "leda", "neutral-female", "orus", "puck",
          "pulcherrima", "rasalgethi", "sadachbia", "sadaltager", "schedar",
          "sulafat", "umbriel", "vindemiatrix", "young-male", "zephyr",
          "zubenelgenubi"]
LONG_VOICES = ["deeper-male", "neutral-female", "young-male", "kore",
               "charon", "sadaltager", "fenrir", "aoede"]


def gen(port, text, voice, seed):
    body = json.dumps({"text": text, "voice": voice, "format": "wav",
                       "seed": seed, "stream": False},
                      ensure_ascii=False).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/tts", data=body,
                                 headers={"Content-Type": "application/json"})
    data = urllib.request.urlopen(req, timeout=1800).read()
    return (len(data) - 44) / 2 / 44100


jobs = [("M", v, ti, s) for v in VOICES for ti in range(len(MEDIUM))
        for s in (3, 11, 42)]
random.Random(13).shuffle(jobs)
jobs = jobs[:550]
longs = [("L", v, ti, s) for v in LONG_VOICES for ti in range(len(LONG))
         for s in (3, 11)]
random.Random(17).shuffle(longs)
jobs += longs[:40]

total, n, fails, t0 = 0.0, 0, 0, time.time()
for kind, v, ti, s in jobs:
    text = MEDIUM[ti] if kind == "M" else LONG[ti]
    try:
        a = gen(8011, text, v, s + 1000 * ti)
        total += a
        n += 1
        fails = 0
        line = (f"[corpus] {n}/{len(jobs)} {kind}{ti} {v} s{s}: {a:.1f}s "
                f"(sum {total / 60:.1f} min, "
                f"wall {(time.time() - t0) / 60:.1f} min)")
        print(line, flush=True)
        if n % 25 == 0:
            print(f"[corpus-mile] {n}/{len(jobs)} takes, "
                  f"{total / 60:.1f} min audio, "
                  f"wall {(time.time() - t0) / 60:.1f} min", flush=True)
    except Exception as e:
        fails += 1
        print(f"[corpus] FAIL {kind}{ti} {v}: {e}", flush=True)
        if fails >= 5:
            print("[corpus] ABORT: 5 consecutive failures", flush=True)
            break
        time.sleep(5)
print(f"[corpus] DONE {n} takes, {total / 60:.1f} min audio, "
      f"wall {(time.time() - t0) / 60:.1f} min", flush=True)
