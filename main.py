import numpy as np
import random as r
import matplotlib.pyplot as plt
from geopy.distance import geodesic
itermax = 100

listcity = ["Вінница", "Дніпропетровськ", "Донецьк", "Житомир", "Запоріжжя", "Івано-Франківськ", "Київ", "Кіровоград", "Луганськ", "Луцк", "Львів", "Миколаїв", "Одеса", "Полтава", "Рівне", "Сімферополь", "Суми", "Тернопіль", "Ужгород", "Харьків", "Херсон", "Хмельницький", "Черкаси","Чернівці", "Чернігів"]

class Node:
    def __init__(self, idn, x, y):
        self.idn = idn
        self.pos = np.array((float(x), float(y)))

coordinates = np.array([
    [28.3, 49.14],
    [35.05, 48.3],
    [37.45, 47.59],
    [28.4, 50.16],
    [35.15, 47.58],
    [24.45, 48.56],
    [30.3, 50.27],
    [32.15, 48.3],
    [39.15, 48.35],
    [25.15, 50.45],
    [24.02, 49.51],
    [32, 46.58],
    [30.45, 46.28],
    [34.84, 49.36],
    [26, 50.35],
    [34.06, 44.58],
    [34.45, 50.53],
    [25.3, 49.34],
    [22.15, 48.38],
    [36.13, 50],
    [32.3, 46.38],
    [27, 49.24],
    [32, 49.27],
    [25.57, 48.17],
    [31.18, 51.29]])

numcity = coordinates.shape[0]
bestpath=np.zeros(numcity)
bpath=np.zeros(numcity)
class Bee:
    def __init__(self):
        self.choosen_nodes = []
        self.recuiter = True
        self.distance = 0.0

    def choose_rand_move(self, move, nods):
        # choosen node must be unique
        for i in range(move):
            if self.is_complete():
                break
            else:
                sel = nods[r.randint(0, len(nodes) - 1)]
                while sel in self.choosen_nodes:
                    sel = nods[r.randint(0, len(nodes) - 1)]
                self.choosen_nodes.append(sel)

            self.total_distance()

    def change_role(self, role):
        self.recuiter = role

    def replace_nodes(self, nods):
        self.choosen_nodes = nods
        self.total_distance()

    def total_distance(self):
        distance = 0.0
        for i in range(len(self.choosen_nodes) - 1):
            node1 = self.choosen_nodes[i]
            node2 = self.choosen_nodes[i + 1]
            distance += geodesic(node1.pos, node2.pos).miles
        distance += geodesic(self.choosen_nodes[-1].pos, self.choosen_nodes[0].pos).miles
        self.distance = distance

    def is_complete(self):
        if len(self.choosen_nodes) >= len(nodes):
            return True
        else:
            return False


def load_nodes(filename):
    ret = []
    with open(filename) as f:
        nodes_s = f.readlines()
    nodes_s = [x.strip() for x in nodes_s]
    for n in nodes_s:
        node = n.split(' ')
        ret.append(Node(node[0], node[1], node[2]))
    return ret


nodes = load_nodes("data1")


# Bee Colony Optimization Algorithm








epoch = 10
n_bee = 10000
n_move = 2

bees = []
best_bee = Bee()

e = 0

    # init bees
for i in range(n_bee):
    bees.append(Bee())


iter=0
dd=10000
bestdd=0

while iter <itermax:

    print("Iteration ",iter)

    if best_bee.is_complete():
        epoch = 10
        n_bee = 500
        n_move = 2

        bees = []
        best_bee = Bee()

        e = 0

        # init bees
        for i in range(n_bee):
            bees.append(Bee())

    while not best_bee.is_complete():
        print("\nEpoch", e + 1)

        print("forward pass")
        # forward pass
        for bee in bees:
            bee.choose_rand_move(n_move, nodes)

        # backward pass
        print("evaluating")
        bees = sorted(bees, key=lambda be: be.distance, reverse=False)
        best_bee = bees[0]

        print("Best distance so far", best_bee.distance)
        print("Best route so far", [n.idn for n in best_bee.choosen_nodes])

        bpath = [int(item) for item in [n.idn for n in best_bee.choosen_nodes]]
        bestdd=best_bee.distance

        print("Bees are making decision to be recruiter or follower")
        Cmax = max(bees, key=lambda b: b.distance).distance
        Cmin = min(bees, key=lambda b: b.distance).distance

        recruiters = []

        for bee in bees:
            Ob = (Cmax - bee.distance) / (Cmax - Cmin)  # range [0,1]
            probs = np.e ** (-(1 - Ob) / (len(bee.choosen_nodes) * 0.01))
            rndm = r.uniform(0, 1)
            # print "ob and probs", Ob, probs
            if rndm < probs:
                bee.change_role(True)
                recruiters.append(bee)
            else:
                bee.change_role(False)

        print("number of recruiter", len(recruiters))
        print("Bees are choosing their recruiter")
        # creating a roulette wheel
        divider = sum([(Cmax - bee.distance) / (Cmax - Cmin) for bee in recruiters])
        probs = [((Cmax - bee.distance) / (Cmax - Cmin)) / divider for bee in recruiters]
        cumulative_probs = [sum(probs[:x + 1]) for x in range(len(probs))]

        for bee in bees:
            if not bee.recuiter:
                rndm = r.uniform(0, 1)
                selected_bee = Bee()
                for i, cp in enumerate(cumulative_probs):
                    if rndm < cp:
                        selected_bee = recruiters[i]
                        break
                bee.replace_nodes(selected_bee.choosen_nodes[:])
        e += 1
    if dd > bestdd:
        dd = bestdd
        bestpath = bpath
        it=iter

    iter=iter+1

axis_font = {'fontname': 'Arial', 'size': '6'}




bestpath=[2,  8, 19, 13, 16,  7, 22, 24,  6,  3,  0, 21, 14,  9, 17,  5, 10, 18, 23, 12, 11, 20, 15,  4,  1,]
plt.xlim([20, 41])
        # xScope
plt.ylim([44, 52])
        # yrange


img = plt.imread(r'C:\Users\bromotdi\Desktop\gk/ukraine-map-coloring-page-hq2.jpg')
fig, ax = plt.subplots()
        # ax.imshow(img, extent=[21.8, 40, 44, 53],alpha=.25, zorder=1)
ax.imshow(img, extent=[21, 41, 43.7, 52.5])

plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='>')
for i in range(numcity - 1):
            # Draw the best path between two cities by coordinates
        m, n = bestpath[i], bestpath[i + 1]
        print("best_path:", m + 1, n + 1, listcity[m], "->", listcity[n],
        geodesic(coordinates[m], coordinates[n]).miles)
        plt.text(coordinates[m][0], coordinates[m][1], listcity[m], **axis_font)
        plt.text(coordinates[n][0], coordinates[n][1], listcity[n], **axis_font)
        plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')

plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[24])][0]],[coordinates[int(bestpath[0])][1], coordinates[int(bestpath[23])][1]], 'b')

        # ax.figure.figimage(im,alpha=.25, zorder=1)
ax = plt.gca()
ax.set_title("Best Path")
ax.set_xlabel('X_axis')
ax.set_ylabel('Y_axis')

plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
plt.close()



