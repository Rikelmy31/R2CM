"""Diagnóstico da string do cubo (ordem U R F D L B, 54 letras) antes de chamar o Kociemba.

O erro "Some edges are undefined" do twophase significa: alguma aresta (peça de 2 cores)
da string lida não existe em um cubo real. As duas causas possíveis são:
  1) um adesivo foi classificado com a cor errada (laranja/vermelho, branco/amarelo...)
  2) uma face foi lida girada/espelhada em relação à orientação esperada (np.rot90 errado)

Uso no main.py, logo antes de sv.solve():
    import cubo_check
    cubo_check.diagnosticar(cube_string)
"""
import itertools

FACES = "URFDLB"

# Posição (x, y, z) de cada adesivo, na ordem de leitura do Kociemba (linha a linha, de fora).
def _pos_facelet(face, i, j):
    if face == 'U': return (j - 1, 1, i - 1), (0, 1, 0)       # topo da imagem = trás
    if face == 'R': return (1, 1 - i, 1 - j), (1, 0, 0)
    if face == 'F': return (j - 1, 1 - i, 1), (0, 0, 1)
    if face == 'D': return (j - 1, -1, 1 - i), (0, -1, 0)
    if face == 'L': return (-1, 1 - i, j - 1), (-1, 0, 0)
    if face == 'B': return (1 - j, 1 - i, -1), (0, 0, -1)


# índice 0..53 -> (posição da peça, normal)
FACELETS = [_pos_facelet(f, i, j) for f in FACES for i in range(3) for j in range(3)]


def _pecas():
    """Agrupa os índices dos adesivos por peça (cantos = 3 adesivos, arestas = 2)."""
    d = {}
    for idx, (pos, _) in enumerate(FACELETS):
        d.setdefault(pos, []).append(idx)
    return [v for v in d.values() if len(v) >= 2]


PECAS = _pecas()
ARESTAS = [p for p in PECAS if len(p) == 2]
CANTOS = [p for p in PECAS if len(p) == 3]

_SOLVED = "".join(f * 9 for f in FACES)
ARESTAS_VALIDAS = {frozenset(_SOLVED[i] for i in p) for p in ARESTAS}
CANTOS_VALIDOS = {frozenset(_SOLVED[i] for i in p) for p in CANTOS}


def problemas(s):
    """Lista de textos descrevendo o que há de errado na string (vazia = peças ok)."""
    out = []
    if len(s) != 54:
        return [f"tamanho {len(s)} (deveria ser 54)"]
    for l in FACES:
        if s.count(l) != 9:
            out.append(f"letra {l} aparece {s.count(l)}x (deveria ser 9)")
    vistas = {}
    for tipo, pecas, validas in (("aresta", ARESTAS, ARESTAS_VALIDAS),
                                 ("canto", CANTOS, CANTOS_VALIDOS)):
        for p in pecas:
            cores = frozenset(s[i] for i in p)
            nome = "-".join(s[i] for i in p)
            onde = ", ".join(f"{FACES[i // 9]}{i % 9 + 1}" for i in p)
            if cores not in validas:
                out.append(f"{tipo} inexistente {nome} (adesivos {onde})")
            elif len(cores) == len(p):
                if (tipo, cores) in vistas:
                    out.append(f"{tipo} repetida {nome} (adesivos {onde})")
                vistas[(tipo, cores)] = True
    return out


def _rot90(face9, k):
    """Gira uma face (lista de 9) k x 90° no sentido horário, como np.rot90(f, -k)."""
    g = [face9[0:3], face9[3:6], face9[6:9]]
    for _ in range(k % 4):
        g = [list(r) for r in zip(*g[::-1])]
    return [c for r in g for c in r]


def procurar_rotacoes(s):
    """Testa girar cada face 0/90/180/270° e devolve as combinações que deixam o cubo válido."""
    faces = [list(s[i * 9:(i + 1) * 9]) for i in range(6)]
    achadas = []
    for ks in itertools.product(range(4), repeat=6):
        t = "".join("".join(_rot90(faces[f], ks[f])) for f in range(6))
        if not problemas(t):
            achadas.append((ks, t))
    return achadas


def diagnosticar(s):
    print("\n--- DIAGNÓSTICO DO CUBO ---")
    for f in range(6):
        b = s[f * 9:(f + 1) * 9]
        print(f"{FACES[f]}: {b[0:3]} / {b[3:6]} / {b[6:9]}")
    ps = problemas(s)
    if not ps:
        print("Todas as arestas e cantos existem. O erro não está nas cores/rotações.")
        return True
    print("Problemas encontrados:")
    for p in ps:
        print("  -", p)
    achadas = procurar_rotacoes(s)
    if achadas:
        print("\nGirando faces (quartos de volta horários) o cubo fica válido:")
        for ks, _ in achadas[:5]:
            print("  ", {FACES[i]: ks[i] for i in range(6) if ks[i]})
        print("=> Uma face está sendo lida na orientação errada: ajuste o np.rot90 dela em Face.update.")
    else:
        print("\nNenhuma rotação de face resolve => há adesivo com COR classificada errada "
              "(veja as peças listadas acima) ou face espelhada.")
    return False
