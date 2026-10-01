from collections import defaultdict
import json
from pathlib import Path
from kanren import Relation, facts, run, conde, var

region = Relation()
match = Relation()
shootout = Relation()
beat = Relation()
champion = Relation()

region_data = {
    "North Korea": "Dong Bac A",
    "United Arab Emirates": "Trung Dong",
    "Palestine": "Trung Dong",
    "Bahrain": "Trung Dong",
    "Bangladesh": "Nam A",
    "Indonesia": "Dong Nam A",
    "Malaysia": "Dong Nam A",
    "Vietnam": "Dong Nam A",
    "Saudi Arabia": "Tay A",
    "Syria": "Tay A",
    "Iran": "Tay A",
    "China": "Dong A",
    "Japan": "Dong A",
    "Hong Kong": "Dong A",
    "South Korea": "Dong A",
    "Uzbekistan": "Trung A"
}

with Path(__file__).with_name("football_relationships.json").open(encoding="utf-8") as f:
    match_data = json.load(f)["matches"]

matches = [
    (
        item["team1"], item["team2"], item["score1"], item["score2"],
        item.get("penalties1"), item.get("penalties2"), item["winner"]
    )
    for item in match_data
]

for team, area in region_data.items():
    facts(region, (team, area))

for t1, t2, s1, s2, p1, p2, winner in matches:
    facts(match, (t1, t2, s1, s2))
    if p1 is not None:
        facts(shootout, (t1, t2, p1, p2))
    loser = t2 if winner == t1 else t1
    facts(beat, (winner, loser))

facts(champion, ("South Korea",))

opponent = var()
goals_for = var()
goals_against = var()
penalties_for = var()
penalties_against = var()
area = var()
team = var()
sea_opponent = var()
nea_opponent = var()

print("a) Cac doi Vietnam da thi dau va ti so:")
vietnam_matches = run(
    0,
    (opponent, goals_for, goals_against),
    conde(
        [match("Vietnam", opponent, goals_for, goals_against)],
        [match(opponent, "Vietnam", goals_against, goals_for)]
    )
)
for opp, gf, ga in vietnam_matches:
    penalties = run(
        1,
        (penalties_for, penalties_against),
        conde(
            [shootout("Vietnam", opp, penalties_for, penalties_against)],
            [shootout(opp, "Vietnam", penalties_against, penalties_for)]
        )
    )
    score = f"Vietnam {gf}-{ga} {opp}"
    if penalties:
        pf, pa = penalties[0]
        score += f" | luan luu {pf}-{pa}"
    print(score)

print("\nb) Nhat Ban da gap cac doi thuoc khu vuc:")
japan_opponents = run(
    0,
    opponent,
    conde(
        [match("Japan", opponent, goals_for, goals_against)],
        [match(opponent, "Japan", goals_against, goals_for)]
    )
)
for opp in dict.fromkeys(japan_opponents):
    areas = run(0, area, region(opp, area))
    print(f"{opp}: {areas[0] if areas else 'Khong ro'}")

print("\nc) Thong ke ban thang / ban thua cua moi doi:")
stats = defaultdict(lambda: [0, 0])
for t1, t2, s1, s2, p1, p2, winner in matches:
    stats[t1][0] += s1
    stats[t1][1] += s2
    stats[t2][0] += s2
    stats[t2][1] += s1
for name in sorted(stats):
    scored, conceded = stats[name]
    print(f"{name}: ghi {scored}, thua {conceded}")

print("\nd) Cac cap dau va ti so:")
for t1, t2, s1, s2, p1, p2, winner in matches:
    if p1 is None:
        print(f"{t1} {s1}-{s2} {t2}")
    else:
        print(f"{t1} {s1}-{s2} {t2} | luan luu {p1}-{p2}")

print("\ne) Doi vo dich:")
print(run(1, team, champion(team))[0])

print("\nf) Doi thang doi khu vuc Dong Nam A va Dong Bac A:")
answer = run(
    0,
    team,
    beat(team, sea_opponent),
    region(sea_opponent, "Dong Nam A"),
    beat(team, nea_opponent),
    region(nea_opponent, "Dong Bac A")
)
print(list(dict.fromkeys(answer)))
