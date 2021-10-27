import numpy as np

def write_genome(gene_groups, fname = 'genome.txt'):
    with open(fname, 'w') as gfile:
        for i in range(len(gene_groups)):
            gfile.write('>g' + str(i) + '\n')
            for g in gene_groups[i]:
                if g[0] == g[-1] and len(g) >= 2:
                    gfile.write('C: ' + ' '.join([str(x) for x in g[:-1]]) + '\n')
                else:
                    gfile.write('L: ' + ' '.join([str(x) for x in g]) + '\n')
            gfile.write('\n')
            
def read_genome(fname = 'genome.txt'):
    gene_groups = []
    with open(fname, 'r') as gfile:        
        gene = []
        tag = False
        
        for gline in gfile:
            if gline.startswith('>'):
                if tag and gene:
                    gene_groups.append(gene)
                tag = True
                gene = []
                continue
            
            genes = gline.strip().split()
            if genes == []:
                continue
            if genes[0] == 'C:':
                genes.append(genes[1])
            elif genes[0] != 'L:':
                continue
            gene.append([int(x) for x in genes[1:]])
        if gene:
            gene_groups.append(gene)
    return gene_groups