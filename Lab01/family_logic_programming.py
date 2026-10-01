import json
from kanren import Relation, facts, run, conde, var, eq

def unique(items):
    return list(dict.fromkeys(items))


def parent(x, y):
    return conde([father(x, y)], [mother(x, y)])


def grandparent(x, y):
    temp = var()
    return conde((parent(x, temp), parent(temp, y)))


def sibling(x, y):
    temp = var()
    return conde((parent(temp, x), parent(temp, y)))


def uncle(x, y):
    temp = var()
    return conde((parent(temp, x), grandparent(temp, y)))


if __name__ == '__main__':
    # Tạo 2 biến cơ sở father và mother
    father = Relation()
    mother = Relation()

    # Nạp data từ file json
    with open('relationships.json', encoding='utf-8') as f:
        d = json.loads(f.read())

    # Tạo sự kiện cho father(x, y)
    for item in d['father']:
        facts(father, (list(item.keys())[0], list(item.values())[0]))

    # Tạo sự kiện cho mother(x, y)
    for item in d['mother']:
        facts(mother, (list(item.keys())[0], list(item.values())[0]))

    x = var()

    name = 'John'
    # Gọi sự kiện father('John', x)
    output = run(0, x, father(name, x))
    print('\nList of ' + name + "'s children:")
    for item in output:
        print(item)

    name = 'William'
    # Gọi sự kiện mother(x, 'William')
    output = run(0, x, mother(x, name))
    print("\nWho is William's mother ?")
    for item in output:
        print(item)

    name = 'Tiffany'
    fathers = set(run(0, x, father(x, name)))
    uncles = [person for person in unique(run(0, x, uncle(x, name)))
              if person not in fathers]
    print("\nList of Tiffany's uncle:")
    for person in uncles:
        print(person)

    print("\nList of all spouses:")
    y = var()
    z = var()
    couples = unique(run(0, (x, y), father(x, z), mother(y, z)))
    for husband, wife in couples:
        print(f"Husband: {husband} --> Wife: {wife}")
