HTTPS_token = 14120
RT_token = 56898
at_token = 137
hashtag_token = 108
gt_token = 135
lt_token = 133
amp_token = 111
question_token = 136
esclamation_token = 106
period_token = 119
two_periods_token = 131
coma_token = 117
dollar_token = 109
period_coma_token = 132
parenthesis_open_token = 113
parenthesis_closed_token = 114
star_token = 115
slash_token = 120
line_token = 118
underscore_token = 168
tilde_token = 198
virgolette_token = 107
square_parenthesis_open_token = 164
square_parenthesis_closed_token = 166
unk_token = 100
rev_question = 756
period_2 = 1882
period_3 = 10097

others_tokens = (11733, 12022)
twitter_internal_link = 188  # "t":188,#twitter links
co_domain_token = 11170  # "co":11170
eight = 129  # "8":129 ???? maybe useful

special_tokens_list = {
    gt_token, lt_token, amp_token,
    esclamation_token, dollar_token,
    parenthesis_open_token, parenthesis_closed_token,
    star_token, line_token, underscore_token,
    tilde_token, virgolette_token, square_parenthesis_open_token,
    square_parenthesis_closed_token, unk_token,
}

NSFW = {'adultcontent', 'adultfilm', 'adultmovie', 'adultvideo', 'adult', 'anal', 'ass', 'bara', 'barelylegal', 'bdsm',
        'bestiality',
        'bisexual', 'bitch', 'blowjob', 'bondage', 'boob', 'boobs', 'boobies', 'boobys', 'booty', 'bound&gagged',
        'boundandgagged', 'gagged',
        'breast', 'breasts', 'bukkake', 'butt', 'cameltoe', 'creampie', 'cock', 'condom', 'cuck', 'cuckold', 'cum',
        'cumshot', 'cunt',
        'deepthraot', 'thraot', 'deapthroat', 'throat', 'deepthraoting', 'thraoting', 'deapthroating', 'throating',
        'deep-thraot', 'thraot', 'deap-throat', 'throat',
        'deep-thraoting', 'deap-throating',
        'deepthraot', 'deapthroat', 'deepthraoting', 'deapthroating', 'dick', 'dildo', 'emetophilia', 'erotic',
        'erotica', 'erection',
        'erections', 'escort', 'facesitting', 'facial', 'felching', 'femdon', 'fetish', 'fisting', 'futanari', 'fuck',
        'fucking', 'fucked',
        'fucks', 'fucker', 'gangbang', 'gapping', 'gay', 'gentlemens club', 'gloryhole', 'gonzo', 'gore',
        'guro', 'handjob',
        'hardon', 'hentai', 'hermaphrodite', 'hiddencamera', 'hump', 'humped', 'humping', 'hustler',
        'incest', 'jerkoff', 'jerkingoff', 'jerk',
        'kink', 'lesbian', 'lewd', 'lolicon ', 'masturbate', 'masturbating', 'masturbation', 'mature', 'mensclub',
        'menstrual', 'menstral', 'menstraul',
        'milf', 'milking', 'naked', 'naughty', 'nude', 'orgasm', 'orgy', 'orgie', 'pearlnecklace', 'pegging', 'penis',
        'penetration', 'playboy',
        'playguy', 'playgirl', 'porn', 'pornography', 'pornstar', 'pov', 'pregnant', 'preggo', 'pubic', 'pussy', 'rape',
        'rimjob', 'scat', 'semen',
        'sex', 'sexual', 'sexy', 'sexting', 'shemale', 'skank', 'slut', 'snuff', 'snuf', 'sperm', 'squirt', 'suck',
        'swapping', 'tit', 'trans',
        'transman', 'transsexual', 'transgender', 'threesome', 'tube8', 'twink', 'upskirt', 'vagina', 'virgin', 'whore',
        'wore', 'xxx', 'yaoi',
        'yif', 'yiff', 'yiffy', 'yuri', 'youporn',
        # added
        'amateur', 'amateurporn', 'voyeur', 'ebony', 'gonewild', 'gw', 'r18', '18', 'القــضـب', 'القََــذف',
        'انتــصــََاب', 'البورنو',
        'إباحي', 'エッチ', 'only fans', 'onlyfans', 'tinhtrùng', 'rule34', 'hentai', 'hentai', 'nsfwart',
        'thicc',
        'エロ', '変態', '触手', 'おっぱい', 'パイズリ', '乳', 'cbt', 'mistress',
        '喜歡記得轉推愛心', '快來跟我互動', 'line群招生中', '單男需付',
        'nsfw', 'nsfl', 'fap'
        }

Kpop = {'kpop', 'k', 'kcon', 'idol', 'comeback', 'mama', 'mnet', 'gda', 'goldendiscaward',
        '골든', '디스크', '시상식',
        '골든디스크시상식', 'gma', 'gaon', 'gaonmusicaward', 'gaon', 'musicaward', 'nct',
        'nct127', 'bts', 'loona', '이달의소녀', 'gfriend', 'blackpink', 'exo', 'mcnd', 'monsta', 'monstax', 'got7',
        'mamamoo',
        'twice', 'ateez', 'big bang', 'bigbang', 'red velvet', 'redvelvet', 'dkb', 'h&d', 'aoa', 'exid',
        'izone',
        'itzy', 'cravity', 'btob', 'craxy', 'cignature', 'playmgirls', '2z', 'clc', 'unvs', 'xenex',
        'daydream', 'wooah',
        'pentagon', 'bandage', 'redsquare', '2nyne', 'trusty',
        # added     'treasure',
        'jhope', 'suga', 'jung', 'kook', 'jungkook', 'bts_twt',
        'enhypen', 'jungwon', 'sunghoon', '콘서트', '제이오원', '트레저', 'kon', 'koncert', '유니', '콘', '유니버스콘서트',
        'kbs', 'shinee', 'hyunjin', 'ヒョンジン', '현진', 'jisoo', '김지수',
        'lovesick', 'lovesickgirls', 'rosé', '로제', 'suyeon', '수연', 'jongin', '김종인',
        'seventeen', 'jennie', 'straykids',
        'ikon', 'boyz', 'woodz', 'ohmygirl', 'wheein', '정휘인', '강다니엘', '샤이니',
        'chanwoo', '정찬우', 'yunhyeong', '크래비티', '세림', '여자아이들', 'gidle',
        'cherrybullet', 'cherry', 'cherryrush', 'cherry', '체리블렛',
        '유현', '인기가요', '원호', 'wonho',
        '에이티즈', '앨런', 'mapofthesoul', 'votebtsglobal',
        }

covid = {'coronavirus', 'covid', 'covid19', 'covid-19', 'virus', 'syndrome', 'respiratory', 'syndrome',
         'sars', 'mers',
         'stayhome', 'distancing',
         'socialdistancing', 'epidemy', 'epidemic', 'pandemy', 'pandemic', 'emergency',
         'emergency', 'lockdown',
         'quarantine', 'isolation', 'incubation', 'wuhan', 'china', 'chinavirus', 'vaccine', 'vaccino', 'vaccinu',
         'vaccinazione', 'vaccination',
         'bat',
         'pangolin', 'pangolino', 'anteaters', 'manis', 'respirator', 'intensive', 'icu', 'fever',
         'cough',
         'antibody', 'antibodies', 'symptoms', 'asymptomatic', 'disease', 'containment', 'herd',
         'immunity', 'herdimmunity', 'vaccin',  # vaccin is ok both for vaccin and vaccine
         'astrazeneca', 'vaksin', 'moderna', 'pfizer', 'astra', 'astrazeneca', 'zeneca', 'johnson',
         'sputnik', 'sputnikv', 'quarant',  # cover more languages
         'covidvaccine', 'disease', 'transmission', 'outbreak',
         # additional languages
         'كوفيد', 'كوفيد-19', 'قناع', 'مصل', 'تلقيح', 'টিকাদান', 'ভেকচিনলোৱাৰ', 'ভেকচিন', 'লোৱাৰ',
         'تحصين', 'لتحصين', 'قناع', 'الكمامة', 'التباعدالاجتماعي', 'التباعد الاجتماعي', 'اللقاح', 'خليك_بالبيت',
         'كورونا', 'ماسك',
         'txertoa', 'txertatzen', 'пандемия', 'ваксинират', 'vacuna', 'coronavac', 'vacunat', 'pneumonie', 'pandemieën',
         'gevaccineerd', 'variant', 'rokot', 'pandemie', 'πνευμονία', 'εμβολι', 'કોરોના', 'માસ્ક', 'રસી', 'להתחס',
         'הקורונה', 'מסכה',
         'टीकाकरण', 'वैक्सीन', 'vakcin', 'coprifuoco', 'curfew', 'trombosi', 'thrombosis', 'nomask', 'novax', 'burioni',
         'bassetti', 'fauci', 'plague',
         'plagueinc', 'コロナウイルス', 'コロナ', 'マスク', 'icu', '肺炎', 'ಲಸಿಕೆ', '코로나',
         'smitt', 'ویکسینیشن', 'aşı', 'bakımda', '武漢', 'inmunización', 'inmuniza', 'vacuna',
         'assembramento',
         'assemblamento', 'coviddi', 'coviddì',  # some mistaken words
         'biontech', 'mrna',  # 'bigpharma',
         '코로나', 'impf', 'ワクチン', 'maske', 'epidemie', 'パンデミック', '감염병',
         '마스크', '백신', '모임', 'versammlung', '搜集', '新冠', '面具',
         '疫苗', 'máscara', '缓冲', '綿棒', '면봉', 'swab', 'hisopo',
         'tupfer', 'écouvillon', 'masquer', 'épidémie', 'pandémie', 'asintomico', 'tampone',
         # maybe  'dpcm',

         }

japaneseTrending = {'peing', '質問箱', '#shindanmaker', 'これでフォロワーさん', '猫', 'sinoalice', 'grandsummoners'}

# remove spaces before counting
virtualYoutube = {'ホロ', 'ホロライブ', 'ライブ', 'ブイチューバー', 'バーチャルユーチューバー', 'vtuber', 'virtualyoutuber', 'hololive',
                  'callillust', 'ollie',
                  'ノエラート', 'shirogane', 'shiroganenoel', 'vshojo',
                  'しらぬえ',
                  '絵クロマンサー', 'マリンのお宝', 'ぺこらーと', '白上フブキ',
                  '大神ミオ', '猫又おかゆ', '戌神ころね', 'ameliart', 'artofashes',
                  'kizuna', 'aichannel', 'aichan', 'kizunaa', 'kizunaai', 'ai-chan', 'kizuna', 'helloworld2020',
                  'themiracle', '「themiracle」', 'oragestar', 'アイちゃん', 'キズナアイ',
                  'gawrt', 'gawrgura', 'がうるぐら', 'がうる', 'がうるぐら', 'ぐら',
                  'inugami', 'korone', '戌神ころね', '戌神', 'ころね'
                                                      'nyanners', 'nyannyanners', '根羽清ココロ', 'nebasei',
                  '桐生ココ', 'cococh', 'kiryu', 'たつのこ', '桐生可可', 'ironmouse',
                  '輝夜月', 'akari', 'ミライアカリ', '猫宮ひなた', '茨ひより', 'hiyori',
                  '夏色まつり', '潤羽るしあ', '樋口楓', '椎名唯華', '兎田ぺこら', 'ぺこら', '兎田',
                  '電脳少女シロ', 'siroch', 'sirochannel', 'もこ田めめめ',
                  '田中ヒメ', '鈴木ヒナ', 'ヒメヒナ', 'ヒナヒメ', 'himehina',
                  'fubuki', 'shirakami', '白上', '白上フブキ',
                  '湊あくあ', '宝鐘マリン', '宝鐘', 'マリン', 'はあちゃま', 'ワトソンアメリア', 'ワトソン', 'アメリア',
                  'ワトソンアメリア', '潤羽', 'るしあ', '猫又おかゆ', '猫又', 'okayu',
                  }

anime = {
    'shonen', 'shonenjump', 'ジャンプ', 'gafes2021', 'アニメ', 'anime', 'マンガ', '漫画', 'manga', 'animé',
    # '오타쿠를','otaku','オタク',
    '遊郭編', '鬼滅の刃遊郭編', '鬼滅の刃', '鬼滅之刃', '鬼滅', 'キメツ', 'yaiba', 'kimetsu', 'demonslayer', '風柱', '風柱', '竈門炭治郎', '竈門', '炭治郎',
    '鬼滅祭', 'nezuko', 'ねずこ', '禰豆子',
    'ワンピース', 'ルフィ', 'luffy', 'ゾロ', 'サンジ', 'sanji', 'hancock', 'ハンコック', 'perona', 'ペローナ', 'nami', 'ナミ', 'mihawk', 'ミホーク',
    'nicorobin', 'ニコロビン', 'ニコロビン', 'chopper', 'チョッパー',  # 'onepiece',
    'bleach', 'ブリーチ', 'ichigo',
    '進撃の巨人', 'shingeki', 'kyojin', 'attackontitan', 'ontitan', 'aot', 'エレンイェーガー', 'エレンイェーガー', 'ミカサアッカーマン', 'ミカサアッカーマン',
    'ミカサ', 'mikasa', 'eren',
    '岸辺露伴は動かない', 'kishiberohan', 'jojo', 'ジョジョ', 'ジョジョの奇妙な冒険', '奇妙な冒険', 'bizzarreadventure', 'jjba', 'jyro',
    'jyrozeppeli',
    'ダンまち', 'ダンジョンに出会いを求めるのは間違っているだろうか', 'danmachi', 'isitwrongtotrytopickupgirlsinadungeon', 'hestia', 'ヘスティア',
    'ベルクラネル', 'ベルクラネル', 'ベルくん', 'ベルきゅん', 'bellcranel', 'リリルカアーデ', 'リリルカアーデ', 'liliruca',
    'goblinslayer', 'ゴブスレ',
    '呪術廻戦', 'じゅじゅ', 'jujutsu', 'jujutsukaisen', '虎杖', 'itadori', '五条', 'gojo', '花御', '東堂', '釘崎', '伏黒', '禅院真希',
    'mushokutensei', 'mushoku_tensei', '無職転生',
    '私に天使が舞い降りた', 'わたてん', 'wataten',
    'reincarnatedasaslime', 'asaslime', '転生したらスライムだった件', '転スラ', 'tensura', 'リムル', 'rimulu', 'rimuru',
    'スラムダンク',  # 'slamdunk', maybe
    'コナン', 'conan', '名探偵コナン',
    'シャーマンキング', 'shamanking',
    '宇宙よりも遠い場所', 'よりもい',
    'ハイキュ', 'haikyu', '月島', 'tsukishima', '翔陽', 'hinata', '日向', 'shoyo', '影山', 'kageyama', '飛雄', 'tobio', '及川',
    'oikawa', 'bokuto',
    'ヒロアカ', 'heroaca', '僕のヒーローアカデミア', 'ヒーローアカデミア', '轟焦凍', '轟焦', 'お茶子', 'ochaco', 'オールマイト', 'エンデヴァー',
    'bokunohero', 'heroacademia', 'myhero', 'myheroacademia', 'deku', 'overhaul', 'midoriya', '出久', 'デク', 'mha', 'bnha',
    'ドラゴンボール', 'dragonball', 'カカロット', 'kakarot', '悟空', 'goku', 'ベジータ', 'vegeta',
    'cellsatwork', 'lavoriincorpo', 'はたらく細胞', '白血球さん', '白血球',  # '赤血球',
    '銀魂', 'gintama', '銀時', 'gintoki',
    'promisedneverland', '約束のネバーランド', 'ネバーランド',
    're:zero', 'rezero', 're:ゼロ', 'reゼロ', '高橋李依', 'エミリア', 'スバルくん', 'レム', 'ラム',
    'saikikusuo', 'saikik', 'saiki', '斉木楠雄', 'さいきくすお',
    'sao', 'swordartonline', 'ソードアートオンライン', 'ソードアート', 'ソードアートオンライン', 'kirito', 'asuna', 'キリト', 'アスナ',
    'yugioh', 'yu-gi-oh', 'yu-gi', 'yugi', '遊戯王',
    'naruto', 'ナルト', 'boruto', 'ボルト', 'gaara',
    'hxh', 'hunterxhunter', 'ハンター', 'ハンター×ハンター', 'ハンターxハンター', 'killua', 'zoldyck',
    'Yuu☆Yuu☆Hakusho', 'yūyūhakusho', 'yuyuhakusho', 'yudeglispettri', '幽☆遊☆白書', '幽遊白書',
    'tokyoghoul', '東京喰種', 'beastars', 'ビースターズ', 'kakegurui', '賭ケグルイ', 'どろろ', 'dororo', 'blackclover', 'ブラッククローバー',
    'anohana', 'あの花', 'あのはな',
    'onepunch', 'onepunchman', 'ワンパンマン', 'deathnote', 'デスノート', 'violetevergarden', 'ヴァイオレットエヴァーガーデン', 'ヴァイオレットエヴァーガーデン',
    'mobpsycho', 'モブサイコ', 'フェアリーテイル', 'fullmetal', 'fulmetal', '鋼の錬金術師',
    'interstella', 'inuyasha', 'sailormoon', 'worldtrigger', 'ワールドトリガー', 'doraemon', 'ドラえもん',
    'digimon', 'デジタルモンスター', 'デジモン',
    'スマイルプリキュア', 'プリキュア', 'ゾンビランドサガ', 'ゾンビランド', '黒子のバスケ',
    '黒子',
}

sports = {'sport', 'basketball', 'football', 'soccer', 'hockey', 'baseball', 'kobe', 'kobebryant', 'bryant',
          'boxing',
          'rip kobe', 'ripkobe', 'nba', 'lakers', 'spurs', 'celtics', 'warriors', 'playoff', 'playoff',
          'nbaplayoff',
          'finals', 'nba', 'nbafinals', 'bball', 'lebron', 'lebron james', 'lebronjames', 'kingjames', 'nfl',
          'nhl',
          'mlb', 'epl', 'premiere', 'league', 'premiereleague', 'seriea', 'liga',
          'league',
          'league1', 'efl', 'süper', 'lig', 'süperlig', 'superlig', 'champions',
          'champions league',
          'espn', 'fox', 'foxnews', 'foxsport', 'sky', 'skysport', 'sport', 'sportnews', 'sportsnews',
          # added
          'badminton', 'archery', 'athletics', 'volleyball', 'golf', 'canoeing', 'hiking', 'handball', 'rollerblading',
          'skating',
          'judo', 'karate', 'netball', 'rowing', 'tennis', 'goalkeeper', 'goalpost', 'linesman', 'guardalinee',
          'jogging',
          'fußball', 'eishockey', 'kegeln', 'segeln', 'reiten', 'rollschuh', 'schlittschuh',
          'radfahren', 'schwimmen', 'joggen', 'wandern', 'windsurfing', 'angeln', 'turnen',
          'bergsteigen', 'ringen', 'tauchen',
          'futbol', 'fútbol', 'béisbol', 'baloncesto', 'voleibol', 'tenis', 'básquetbol', 'bolos', 'boxeo', 'canotaje',
          'championship', 'campeonato', 'porrista', 'alpinismo', 'entrenador', 'competencia', 'críquet',
          'esgrima', 'patinaje',
          'うんどう', 'スポーツ', 'やきゅう', 'すもう', 'じゅうどう', 'けんどう', 'あいきどう',
          'からて', 'すいえい', 'たいそう', 'じょうば', 'しゃげき', 'たっきゅう',
          'ゴルフ', 'テニス', 'サッカー', 'バスケットボール', 'バスケット', 'バスケ', 'バレーボール',
          'フットボール', 'アメリカンフットボール', 'アメフト',
          'ラグビー', 'しょうぎ', 'ご', 'ソフトボール', 'きゅうどう', 'せんしゅ',
          '運動', '体育', '野球', '相撲', '柔道', '剣道', '合気道',
          '空手', '水泳', '体操', '卓球', '将棋', '碁', '弓道', '選手',

          '足球', '篮球', '棒球', '射箭', '游泳', '自行', '自行车',
          '拳击', '曲棍球', '乒乓球', '柔道', '皮划艇', '网球',
          '排球', '跳水', '蹦床', '滑雪', '冲浪', '高尔夫',

          '야구', '농구', '비치발리', '복싱', '권투', '펜싱', '축구',
          '하키', '유도', '리듬체조', '조정', '소프트볼', '탁구', '태권도',
          '레슬링', '골프', '검도', '씨름',

          'arbitre',

          'क्रिकेट', 'खेल', 'क्रिकेटर',

          'qb', 'quarter', 'quarterback', 'referee',
          'brady', 'bowl', 'homerun', 'wheeler', 'ufc', 'sports', 'easports',
          'denard', 'robinson', 'ncaa',  # this may be better as a videogame{or both}
          'serena', 'williams', 'formula', 'formulaone', 'formula1', 'formula1', 'rugby', 'kagiso', 'rabada',
          'bafana', 'naomi', 'naomiosaka',
          'naismith', 'indvsaus', 'ausvsind', 'livescore',
          'muzi', 'manyike', 'thembinkosi', 'siphesihle', 'ndlovu', 'orlando', 'pirates', 'steven', 'gerrard',
          'bongani', 'zungu',
          'nathan', 'patterson', 'calvin', 'bassey', 'dapo', 'mebude', 'brian', 'kinnear', 'antwerp',
          'europaleague', 'rangers',  # maybe remove this
          'rangersfc', 'uel', 'dundee', 'kris', 'boyd', 'bruno', 'fernandes', 'premier', 'sportsnews',
          'westham', 'jesselingard', 'lingard',
          'chelsea', 'frank', 'lampard', 'franklampard', 'mauricio', 'pochettino', 'ozil', 'mesut', 'manchester',
          'united', 'southampton',
          'mike', 'dean', 'erling', 'haaland', 'borussia', 'dortmund', 'josef', 'bican', 'ronaldo', 'tomas', 'soucek',
          'jan', 'bednarek',
          'michael', 'oliver',
          'dijk', 'andre', 'onana', 'uefa', 'doping', 'nfl', 'bobby', 'boucher', 'rich', 'ohrnberger',
          'ohrnberger', 'antman',
          'athlete', 'zlatan', 'jalen', 'jalenhurts', 'cam', 'newton', 'bowl', 'superbowl', 'lebron',
          'tiger', 'woods',
          'jamal', 'murray', 'kevin', 'garnett', 'steven', 'adams', 'bron', 'grant', 'holloway', 'percy', 'harvin',
          'kingjames',
          'jayson', 'tatum', 'kevin', 'durant', 'bronny', 'olympics',
          'florentino', 'fichalo', 'fútbol',
          'calcio', 'arbitro', 'tifosi', 'tifoso', 'sci', 'sciare', 'sciistico', 'sciistica',
          'sportivo', 'sportiva', 'ciclismo', 'pallavolo',
          'indvseng', 'engvsind', 'bowling', 'cricket', 'jackleach', 'jimmy9', 'bumblecricket',
          'nassercricket', 'athersmike', 'robkey612', 'nhl',
          # 'coach',' arsenal ','head coach'
          # from trending

          'nfl', 'fulham', 'packers', 'ufc', 'brasileirão', 'palmeiras', 'psg', 'บอล', 'liverpool',
          'patrickmahomes', 'mshomes', 'bayern', 'santos', 'botafogo', 'الاهلي', 'palmeiras', 'corinthians',
          'alcoyano', 'flamengo', 'كورنيلا', 'برشلونة', 'aaron', 'jra', 'alavés', 'mcgregor', 'grêmio', 'gremio',
          'brady',
          'tombrady', 'poirier', 'dustinpoirier', '위컴', '토트넘', 'wycombe', 'tottenham', 'lampard', 'inter', 'juventus',
          'galatasaray',
          'bundesliga', 'isl', 'trabzonspor', 'الاهلي', 'fenerbahçe', 'gronkowski', '에버턴', '토트넘', 'cricinfo',

          'staley', 'mahomes', 'yreek', 'goff', 'jaredgoff', 'jjwatt',
          # maybe 'browns',

          }

slang = {
    'pog', 'lmao', 'lol', 'btw', 'tbh', 'idgaf', 'afaik',
    'brb', 'btaim', 'dae', 'dyk', 'eli5', 'fbf', 'fbo', 'ff', 'fomo',
    'ftfy', 'ftw', 'fyi', 'g2g', 'gtg', 'gg', 'gtr', 'hbd',
    'hifw', 'hmb', 'hmu', 'ht', 'hth', 'icymi', 'idc', 'idk',
    'ikr', 'ily', 'imho', 'imo', 'irl', 'jk', 'lmao', 'lmk', 'lms', 'lol', 'mcm',
    'mfw', 'mtbwy', 'nbd', 'nm', 'nvm', 'omw', 'ootd', 'op', 'ppl', 'rofl', 'roflmao', 'rtfa',
    'sfw', 'sjw', 'smh', 'smdh', 'tbh', 'tbbh', 'tbt', 'tfw', 'tgif', 'tldr',  'tmi', 'wbu', 'wbw', 'wfh', 'wip', 'yolo',
}

myanmar = {'hearthevoiceofmyanmar', 'myanmar', 'respectourvotes',
           'weneeddemocracy', 'wewantjustice', 'helpmyanmar', 'rejectmilitarycoup', 'againstmilitarycoup',
           'militarycoup',
           'feb14coup', 'ม็อบ13กุมภา', 'whatshappeninginmyanmar', 'feb15coup',

           }

genshinImpact = {'genshinimpact', 'genshinimpactfanart', 'genshin', 'xianyu', '原神', '원신',
                 'chilumi', 'zhongli', 'xiangling', 'keqing', 'ningguang', 'mona', 'モナ',
                 'childe', 'tartali', 'タル鍾', 'albedo', 'fischl', 'sucrose', 'bennet',
                 'ganyu', '甘雨', 'xingqiu', 'xiao', 'diona', 'beidou', 'xinyan',
                 'albedogeo',
                 }

nintendogames = {'nintendoswitch', 'animalcrossing', 'animalcrossing', 'acnh', 'acnl', 'pokemon',
                 'pokèmon',
                 'pokémon', 'supermario', 'super mario', 'minecreaft', 'splatoon', 'zelda',
                 'breathofthewild', 'xenoblade', 'nintendo',
                 'mario kart', 'mariokart', 'mk8d', 'smashbros', 'supersmashbros', 'bowserfury',
                 'bowser fury', 'masuda',
                 'ring fit advanture', 'ringfitadventure',
                 'ニンテンドースイッチ', 'ニンテンドー', 'どうぶつの森', 'どうぶつ', 'ポケモン', 'ポケットモンスター',
                 'スーパーマリオ', 'スーパー マリオ', 'ゼルダ', 'スプラトゥーン', 'ゼノブレイド', 'ファイナルファンタジー',
                 'マリオカート', 'マリオ カート', 'スマブラ', '増田', 'リングフィット アドベンチャ', 'リングフィットアドベンチャ',
                 'nintendodirect',
                 'aceattorney', '逆転裁判', 'wii',
                 }
# TODO add something if this can help
multiconsolegames = {'persona5', 'p5', 'ff', 'final fantasy', 'finalfantasy', 'monsterhunter',
                     'monster hunter', 'ペルソナ',
                     'ファイナルファンタジー', 'モンスターハンター',
                     }

games = {'mxgp', 'hitman3', 'ride4',
         'redout', 'cybershadow', 'stronghold:warlords',
         'アマングアス',  # 'amoungus',
         'ryza', 'encodya', 'helltaker', 'codzombies', 'codblack', 'callofduty',
         'nba2k21', 'pubg', 'pmjl',
         'minecraft', 'roblox', 'bloxyaward',
         'fortnite', 'candy crush', 'undertale', 'deltarune', 'omori',
         # add
         'codwarzone', 'bf5', 'bf6', 'RE8', 'codbo', 'acvalhalla'
         }
# games => FGO =>{tweets with over 30k retweets}=> maybe better one separate category


# add other consoles
crypto = {'musk', 'elon', 'crypto',
          'dogecoin', 'bitcoin', 'btc', 'ethereum', 'eth', 'shitcoin', 'nft', 'earth2',
          'litecoin', 'tothemoon'
          }

love = {
    'valentine', 'valentinesday', 'valentynsdag', 'ਵੇਲੇਂਟਾਇਨ', 'ਵੇਲੇਂਟਾਇਨਡੇ', 'valentinit',
    'የፍቅርቀን', 'የፍቅር', 'valentínusardagur', 'الحب', 'عيدالحب', 'Վալենտինի', 'valentine',
    'sevgililər', 'заљубљених', 'ভালবাসা', 'revalentine', 'バレンタインデー', 'バレンタイン',
    'ويلنٽائن', 'bалянціна', 'valentinovo', 'valentín', 'алентин',
    'ချစ်သူများနေ့', '발렌타인', '情人节', 'evîndaran', 'paléntin', '情人節',
    'valentinu', 'ວັນແຫ່ງຄວາມຮັກ', 'hjärtans', 'valentýn',
    'valentīndiena', 'valentins', 'valentijnsdag', 'vältesdag',
    'วันวาเลนไทน์', 'sevgililer', 'söýgüliler', 'puso', 'bалентина',
    'ystävänpäivä', 'valentinu', 'saint-valentin', 'saintvalentin', 'valentin',
    'Ვალენტინობა', 'tình', 'valentinstag', 'bαλεντίνου', 'valentinsdag',
    'walentynki', 'sanvalentino', 'vday', 'valentinesday', 'happyvalentinesday', 'valentinesday2021',

    'affection', 'appreciation', 'fondness', 'friendship', 'infatutation', 'respect', 'love',
    'tenderness', 'adore', 'adored', 'beloved', 'cherished', 'amiable', 'dear', 'caring', 'faithful',
    'generous', 'romantic', 'passionate', 'amorous', 'kiss',

    'amor', 'cario', 'quiero', 'enamorado', 'beso', 'besar', 'amar', 'querer', 'novio', 'novia',
    'fiancé', 'fiancée', 'fiance', 'fiancee', 'compañero', 'compañera', 'cariño',

    'amour', 'amitié', 'chéri', 'chérie', 'aime', 'baiser', 'bisou',
    'fiançailles', 'fiancer', 'câlin',

    'liebe', 'freund', 'freundin', 'lieben', 'kuss', 'umarmen', 'romantik',
    'süß',

    'abbraccio', 'abbraccia', 'abbracciare', 'amore', 'dolce', 'fidanzato', 'fidanzata',

    '彼女', '彼氏', 'バラ', '愛する', '愛', 'チョコレート', 'ロマンス', 'ロマンチック',
    '抱く', 'キス', 'デート', 'ハート', 'バレンタインカード', '花束', 'キューピッド',

    '吻', '女朋友', '男朋友', '甜', '拥抱', '恋爱', '丘比特', '约会', '情人节贺卡',

    '연애', '여자친구', '남자친구ì', '키스', '안아주다', '사랑', '꽃다발', '데이트', '큐피드',
    '하트',

}

most_searched = {
    'biden', 'mlkday', 'mlk', 'blacklivesmatter',
    'kanyewest', 'kanye', 'boris', 'kamala', 'harris',
    'taylor', 'bbb21', 'bbb', 'redebbb', 'penteado', 'teamgil', 'blackhistorymonth', 'wandavision', 'cricbuzz',
    'หวย', 'spector', 'livinho', 'giriş', 'wayne', 'gaga', 'gorman', 'harris', 'psaki',
    '今泉佑唯', 'imaizumi', 'lovato', 'siwa', 'yoo-jung', 'yoojung', 'kobe', 'kobebryant', 'bryant', 'tubman',
    'wallstreetbets', 'amc', 'stock', 'مهرداد', 'میناوند', 'efraín', 'ruales', 'efraínruales',
    'satta', 'wong-chu', 'wongchu', 'robinhood', 'robinhoodapp', 'moore', 'leachman',
    'leachman', 'cicelytyson', 'cicely', 'tyson', 'dogecoin', 'aöf', 'ชนะ',
    'manson', 'wallen', 'draghi', 'plummer', 'rajivkapoor', 'rajiv', 'kapoor', 'kasia', 'lenhardt', 'kasialenhardt',
    'whedon',
    'pasternak', 'maría', 'grever', 'maríagrever', 'ginacarano', 'carano', 'hirsch', 'fredyhirsch', 'neujahr', '地震',
    'menem',
    'parton', 'inaugurationday', 'perry', 'katyperry', 'presidentbiden', 'inauguration2021',
    'larryking', 'stocks', 'tyson', 'dustindiamond', 'holbrook', 'punxsutawney', 'songz', 'treysongz',
    'dobbs', 'impeachment', 'castor', 'brucecastor', 'acquit', 'acquitted', 'jackson',
    # sports
    'nfl', 'fulham', 'packers', 'ufc', 'brasileirão', 'palmeiras', 'psg', 'บอล', 'liverpool',
    'patrickmahomes', 'mshomes', 'bayern', 'santos', 'botafogo', 'الاهلي', 'palmeiras', 'corinthians',
    'alcoyano', 'flamengo', 'كورنيلا', 'برشلونة', 'aaron', 'jra', 'alavés', 'mcgregor', 'grêmio', 'gremio', 'brady',
    'tombrady', 'poirier', 'dustinpoirier', '위컴', '토트넘', 'wycombe', 'tottenham', 'lampard', 'inter', 'juventus',
    'galatasaray',
    'bundesliga', 'isl', 'trabzonspor', 'الاهلي', 'fenerbahçe', 'gronkowski', '에버턴', '토트넘', 'cricinfo',
    'staley', 'mahomes', 'yreek', 'goff', 'jaredgoff', 'jjwatt',

    # valentine related ok?
    'valentine', 'valentinesday', 'valentynsdag', 'ਵੇਲੇਂਟਾਇਨ', 'ਵੇਲੇਂਟਾਇਨਡੇ', 'valentinit',
    'የፍቅርቀን', 'የፍቅር', 'valentínusardagur', 'الحب', 'عيدالحب', 'Վալենտինի', 'valentine',
    'sevgililər', 'заљубљених', 'ভালবাসা', 'revalentine', 'バレンタインデー', 'バレンタイン',
    'ويلنٽائن', 'bалянціна', 'valentinovo', 'valentín', 'алентин',
    'ချစ်သူများနေ့', '발렌타인', '情人节', 'evîndaran', 'paléntin', '情人節',
    'valentinu', 'ວັນແຫ່ງຄວາມຮັກ', 'hjärtans', 'valentýn',
    'valentīndiena', 'valentins', 'valentijnsdag', 'vältesdag',
    'วันวาเลนไทน์', 'sevgililer', 'söýgüliler', 'puso', 'bалентина',
    'ystävänpäivä', 'valentinu', 'saint-valentin', 'saintvalentin', 'valentin',
    'Ვალენტინობა', 'tình', 'valentinstag', 'bαλεντίνου', 'valentinsdag',
    'walentynki', 'sanvalentino', 'vday',

    # maybe'winx','sophie'(after 30 january),'Dustin Diamond',

}
news = {
    'airstrike', 'texasstorm', 'texas', 'mandates', 'trump',
    'biden', 'wallace', 'limbaugh', 'cuomo', 'janicedean', 'dean',
    'biologicalmales', 'biological', 'maximewaters', 'maxime',
    'johnkerry', 'kerry', 'impeachment', 'kendalljennery',
    'bobby', 'shmurda', 'breaking', 'capitol',
    'kim', 'mitch', 'mcconnell', 'rudy', 'giuliani', 'myanmar',
    'scoop', 'harris', 'leachman', '速報', '地震',

}

# TODO add words
mostPopular = {  # arab
    'رَبُّهُ', 'الله', 'آدم',  # bangladesh "ঈশ্বর"

}

socials = {'instagram', 'facebook', 'twitch', 'tiktok', 'tik tok', 'reddit', 'whatsapp', 'telegram', 'live', 'ライブ',
           'youtube', 'ニコニコ動画', 'linkedin', 'twitter',
           'pinterest', 'snapchat',
           # 'line', this must be uppercase

           }

# TODO think here
# other={"lgbtq",}
# americaPolitics = {'biden','trump','capitolriot','republican','democrat','ransack','military'}
