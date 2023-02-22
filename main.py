import argparse
import logging
import json

import numpy as np
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.manifolds.poincareball import PoincareBall

import hyptorch.pmath as pmath
import hyptorch.nn as pnn


from htsne_impl import TSNE as hTSNE
from sklearn.manifold import TSNE

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


c = 1.0

def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = torch.normal(mean=0.0, std=1.0, size=(dimension, num_points))
    random_directions /= torch.norm(random_directions, dim=0)
    
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = torch.rand(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def generate_riemannian_distri(idx, batch=10, dim=2, scale=1., all_loc=[]):
    
    pball = PoincareBall(dim, c=1)

    loc = random_ball(1, dim, radius=0.999) #ball상의 임의의 점을 return  #shape [1,5]

    if idx == 0:
        loc = torch.zeros_like(loc)

    distri = RiemannianNormal(loc, torch.ones((1,1)) * scale, pball)

    return distri, loc


def generate_riemannian_clusters(clusters=5, batch=20, dim=2, scale=1.):
    embs  = torch.zeros((0, dim))
    means = torch.zeros((0, dim))
    
    pball = PoincareBall(dim, c=1)


    all_loc = []

    labels= []
    

    for i in range(clusters):

        distri, mean = generate_riemannian_distri(idx = i, batch=batch, dim=dim, scale=scale, all_loc=all_loc)
        labels.extend([i] * batch)

        for _ in range(batch):
            embs = torch.cat((embs, distri.sample()[0]))

        means = torch.cat((means, mean))

    ###############################################

    return embs, labels, means


def generate_high_dims():

    embs, labels, means = generate_riemannian_clusters(clusters=5, batch=20, dim=2, scale=0.25)

    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    seed_colors = ['black', 'red', 'b', 'g', 'c']
    # seed_colors = np.random.rand(5,3)
    colors = []
    for label in labels:
        colors.append(seed_colors[label])

    plt.scatter(embs[:,0], embs[:,1], c=colors, alpha=0.3)


    mcolors = []
    for i in range(means.shape[0]):
        mcolors.append(seed_colors[i])

    plt.scatter(means[:,0], means[:,1], c=mcolors, marker='x', s=50)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)
    #####################################################


    plt.savefig("./saved_figures/high_dim" + ".png", bbox_inches='tight', dpi=fig.dpi)

    return embs, colors



def run_TSNE(embeddings, learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0, n_iter=1000):

    tsne = TSNE(n_components=2, method='exact', perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=1)

    tsne_embeddings = tsne.fit_transform(embeddings)

    print ("\n\n")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()


    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)


    _htsne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=0.0, student_t_gamma=1.0, learning_rate=learning_rate, n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration)

    HT_SNE_embeddings = _htsne.fit_transform(dists, embeddings)


    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding



def plot_low_dims(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma):



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30, label=colors)
    # plt.legend()
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/" + "tsne.eps", bbox_inches='tight', dpi=fig.dpi)
    plt.savefig("./saved_figures/" + "tsne.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/" + "HT-SNE.eps", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=30)
    ax.set_aspect('equal')
    plt.axis('off')

    plt.savefig("./saved_figures/" + "CO-SNE.eps", bbox_inches='tight', dpi=fig.dpi)
    plt.savefig("./saved_figures/" + "CO-SNE.png", bbox_inches='tight', dpi=fig.dpi)


def load_embeddings(dataset, dim): #dim : embedding불러올 dimension
    #return : emb, label, mean
    # embedding불러
    #hypernym action embedding index 확인 
    # index으로 임베딩 불러오기 
    #먼저 데이터를 확인
    # dataset is either 50salad or breakfast
    if dataset == '50salad':
        base_path = '/datasets/50salads/hyperbolic_embedding/embedding_groundTruth/' 
        # 필요한 action json을 불러와서 아래 index을 만들자
        # {hyp : [actions]} hierarchy json
        # mapping json 
        #{act : hyp} -> {hyp : [act]} 이렇게 만들고 mapping하면 됨 
        with open('/datasets/50salads/mappings/action_hypernym.json', 'r') as f:
            act_hyp_base = json.load(f)
        with open('/datasets/50salads/mappings/hypernyms.json', 'r') as f:  # {hypact : num}
            hyp_num = json.load(f)
        with open('/datasets/50salads/mappings/actions.json', 'r') as f: # {act : num}
            act_num = json.load(f)

        #load embedding_file
        hyp_emb = np.load(base_path + '/hypernym_50salad_poincare_{}D_.npy'.format(dim))
        act_emb = np.load(base_path + '/action_50salad_poincare_{}D_.npy'.format(dim))

        #change to tensor
        hyp_emb = torch.from_numpy(hyp_emb)
        act_emb = torch.from_numpy(act_emb)

        # make {hyp : [act]}
        act_hyp = dict()
        for hyps in hyp_num.keys():
            act_hyp[hyps] = []
        for act, hyp in act_hyp_base.items():
            act_hyp[hyp].append(act)
        
        # act_hyp : {hyp : [act...]}
        #이걸 숫자로 번호로 mapping
        act_hyp_num = {}
        for hyp, acts in act_hyp.items():
            hyp_number = hyp_num[hyp]
            act_list = []
            for act in acts:
                act_list.append(act_num[act])
            act_hyp_num[hyp_number] = act_list
        
        #act_hyp_num : {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [10, 11, 12, 13, 14], 2: [15, 16]}
        
    elif dataset == 'breakfast':
        base_path = '/datasets/breakfast/hyperbolic_embedding/embedding_groundTruth/'
        #breakfast 구현 필요 
        with open('/datasets/breakfast/mappings/hypernyms.json', 'r') as f:  # {hypact : num}
            hyp_num = json.load(f)
        with open('/datasets/breakfast/mappings/actions.json', 'r') as f: # {act : num}
            act_num = json.load(f)
        
        #load embedding_file
        hyp_emb = np.load(base_path + '/hypernym_breakfast_poincare_{}D_.npy'.format(dim))
        act_emb = np.load(base_path + '/action_breakfast_poincare_{}D_.npy'.format(dim))

        #change to tensor
        hyp_emb = torch.from_numpy(hyp_emb)
        act_emb = torch.from_numpy(act_emb)

        # make {hyp : [act]}
        # act_hyp = dict()
        # for hyps in hyp_num.keys():
        #     act_hyp[hyps] = []
        # for act, hyp in act_hyp_base.items():
        #     act_hyp[hyp].append(act)
        with open('/datasets/breakfast/description_code/breakfast_hierarchy_refine.json', 'r') as f:
            act_hyp = json.load(f)
        
    
        # act_hyp : {hyp : [act...]}
        #이걸 숫자로 번호로 mapping
        act_hyp_num = {}
        for hyp, acts in act_hyp.items():
            hyp_number = hyp_num[hyp]
            act_list = []
            for act in acts:
                act_list.append(act_num[act])
            act_hyp_num[hyp_number] = act_list

    else:
        raise "not Implemented dataset"
    # {1 : [0, 11], 2 : [4,5]} -> 이런식으로 mapping되게끔 만든다 {hyp : [act]}
    
    #각 cluster마다 embedding을 가지고 오고, 그 사이즈만큼 label을 만들고 mean을 계산 
    #cluster수를 define
    num_cluster = len(act_hyp_num.keys())

    embs = torch.zeros((0, dim))
    means = torch.zeros((0, dim))
    label = []

    for cluster in act_hyp_num.keys():
        # label extend은 다른 cluster넘어가기 전에 때려야함
        hyper_emb = torch.index_select(hyp_emb, 0, torch.tensor(cluster))
        embs = torch.cat((embs, hyper_emb))
        
        #action embedding골라오기
        act_index = torch.tensor(act_hyp_num[cluster])

        action_emb = torch.index_select(act_emb, 0, act_index)
        embs = torch.cat((embs, action_emb)) #[11, 512] [S, D]

        #label update
        cluster_label = [cluster] * (len(act_index) + 1) #hypernym(cluster)도 포함
        label.extend(cluster_label)

        #mean update
        mean = torch.mean(embs, 0).unsqueeze(0)

        means = torch.cat((means, mean))

    #embs.shape : [20, 512]
    #means.shape : [3, 512] [cluster num, D]
    #len(label) : 20
    return embs, label, means


def generate_input_embs(dataset, dim):

    embs, labels, means = load_embeddings(dataset, dim)

    #그냥 임베딩 자체를 불러와보자 
    base_path = '/datasets/breakfast/hyperbolic_embedding/embedding_groundTruth/'
    hyp_emb = np.load(base_path + '/hypernym_breakfast_poincare_{}D_.npy'.format(dim))
    act_emb = np.load(base_path + '/action_breakfast_poincare_{}D_.npy'.format(dim))

    hyp_emb = torch.from_numpy(hyp_emb)
    act_emb = torch.from_numpy(act_emb)

    embs = torch.zeros((0, dim))
    embs = torch.cat((embs, hyp_emb))
    embs = torch.cat((embs, act_emb))

    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    #salad은 3개니까..
    if dataset == '50salad':
        seed_colors = ['r', 'g', 'b']
    elif dataset == 'breakfast':
        seed_colors = np.random.rand(47, 3) #TODO make flexible
    else:
        raise "Not implemented"
    # seed_colors = np.random.rand(5,3)
    colors = []
    # for label in labels:
    #     colors.append(seed_colors[label])

    # plt.scatter(embs[:,0], embs[:,1], c=colors, alpha=0.3)
    plt.scatter(embs[:,0], embs[:,1], alpha=0.3)


    # # mcolors = []
    # # for i in range(means.shape[0]):
    # #     mcolors.append(seed_colors[i])

    # # plt.scatter(means[:,0], means[:,1], c=mcolors, marker='x', s=50)

    # circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    # plt.gca().add_patch(circle1)
    # #####################################################


    # plt.savefig("./saved_figures/{}/origin".format(dataset) + ".png", bbox_inches='tight', dpi=fig.dpi)

    return embs, colors

    
def plot_hypertsne(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma, dataset):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    
    # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30, label=colors)

    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.savefig("./saved_figures/{}/".format(dataset) + "tsne.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/{}/".format(dataset) + "HT-SNE.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=30)
    plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], s=30)
    
    #use for breakfast
    # acts = ['pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']
    # for i, txt in enumerate(acts):
    #     plt.annotate(txt, (CO_SNE_embedding[i,0], CO_SNE_embedding[i,1]))


    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/{}/".format(dataset) + "CO-SNE.png", bbox_inches='tight', dpi=fig.dpi)


def generate_compositional(dim):
    base_path = '/datasets/50salads/hyperbolic_embedding/embedding_groundTruth/' 
    #load embedding_file
    hyp_emb = np.load(base_path + '/hypernym_50salad_poincare_{}D_.npy'.format(dim))
    act_emb = np.load(base_path + '/action_50salad_poincare_{}D_.npy'.format(dim))

    #change to tensor
    hyp_emb = torch.from_numpy(hyp_emb)
    act_emb = torch.from_numpy(act_emb)

    embs = torch.zeros((0, dim))
    embs = torch.cat((embs, hyp_emb))
    embs = torch.cat((embs, act_emb))

    print(embs.shape)

    base_path = '/datasets/breakfast/hyperbolic_embedding/embedding_groundTruth/'
    hyp_emb = np.load(base_path + '/hypernym_breakfast_poincare_{}D_.npy'.format(dim))
    act_emb = np.load(base_path + '/action_breakfast_poincare_{}D_.npy'.format(dim))

    hyp_emb = torch.from_numpy(hyp_emb)
    act_emb = torch.from_numpy(act_emb)


    embs = torch.cat((embs, hyp_emb))
    embs = torch.cat((embs, act_emb))

    return embs

def compositional_plot(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    
    # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30, label=colors)

    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.savefig("./saved_figures/{}/".format(dataset) + "tsne.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/composition/" + "HT-SNE.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=30)
    plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], s=30)
    
    #use for breakfast
    # acts = ['pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']
    # for i, txt in enumerate(acts):
    #     plt.annotate(txt, (CO_SNE_embedding[i,0], CO_SNE_embedding[i,1]))

    #use for compositional
    acts = ['cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad', 'peel_cucumber', 'cut_cucumber', 'place_cucumber_into_bowl', 'cut_tomato', 'place_tomato_into_bowl', 'cut_cheese', 'place_cheese_into_bowl', 'cut_lettuce', 'place_lettuce_into_bowl', 'mix_ingredients', 'add_oil', 'add_vinegar', 'add_salt', 'add_pepper', 'mix_dressing', 'serve_salad_onto_plate', 'add_dressing', 'pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']
    for i, txt in enumerate(acts):
        plt.annotate(txt, (CO_SNE_embedding[i,0], CO_SNE_embedding[i,1]))

    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/composition/" + "CO-SNE.png", bbox_inches='tight', dpi=fig.dpi)


def generate_salad(dim):
    base_path = '/datasets/50salads/hyperbolic_embedding/embedding_groundTruth/' 
    #load embedding_file
    hyp_emb = np.load(base_path + '/hypernym_50salad_poincare_{}D_.npy'.format(dim))
    act_emb = np.load(base_path + '/action_50salad_poincare_{}D_.npy'.format(dim))

    #change to tensor
    hyp_emb = torch.from_numpy(hyp_emb)
    act_emb = torch.from_numpy(act_emb)

    embs = torch.zeros((0, dim))
    embs = torch.cat((embs, hyp_emb))
    embs = torch.cat((embs, act_emb))


    return embs

def salad_plot(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    
    # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30, label=colors)

    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.savefig("./saved_figures/{}/".format(dataset) + "tsne.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], s=30)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/50salad/" + "HT-SNE.png", bbox_inches='tight', dpi=fig.dpi)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    # plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=30)
    plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], s=30)
    
    #use for breakfast
    # acts = ['pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']
    # for i, txt in enumerate(acts):
    #     plt.annotate(txt, (CO_SNE_embedding[i,0], CO_SNE_embedding[i,1]))

    #use for compositional
    acts = ['cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad', 'peel_cucumber', 'cut_cucumber', 'place_cucumber_into_bowl', 'cut_tomato', 'place_tomato_into_bowl', 'cut_cheese', 'place_cheese_into_bowl', 'cut_lettuce', 'place_lettuce_into_bowl', 'mix_ingredients', 'add_oil', 'add_vinegar', 'add_salt', 'add_pepper', 'mix_dressing', 'serve_salad_onto_plate', 'add_dressing']
    for i, txt in enumerate(acts):
        plt.annotate(txt, (CO_SNE_embedding[i,0], CO_SNE_embedding[i,1]))

    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig("./saved_figures/50salad/" + "CO-SNE.png", bbox_inches='tight', dpi=fig.dpi)


def generate_embeddings(dataset, dim):
    #dataset 명시시 해당 데이터, action 가져와서 임베딩 만들고
    #저장하는 위치는 우선 COSNE내부에 만들기 위해 여기에 만들지 않음
    if dataset == '50salad':
        base_path = '/datasets/{}/hyperbolic_embedding/embedding_groundTruth/'.format(dataset + 's')
    else:
        base_path = '/datasets/{}/hyperbolic_embedding/embedding_groundTruth/'.format(dataset)

    #load embedding_file
    hyp_emb = np.load(base_path + '/hypernym_{}_poincare_{}D_.npy'.format(dataset, dim))
    act_emb = np.load(base_path + '/action_{}_poincare_{}D_.npy'.format(dataset, dim))

    #change to tensor
    hyp_emb = torch.from_numpy(hyp_emb)
    act_emb = torch.from_numpy(act_emb)

    embs = torch.zeros((0, dim))
    embs = torch.cat((embs, hyp_emb))
    embs = torch.cat((embs, act_emb)) #[hyp_num + act_num, embedded_dim]

    return embs

def run_COSNE(embeddings, n_components=2,learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0, n_iter=1000):
    #run and save COSNE embeddings
    co_sne = hTSNE(n_components, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=n_iter, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()

    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)

    return CO_SNE_embedding

def save_COSNE(embeddings, dataset):
    save_emb_path = '/workspace/CO-SNE/save_embedding/' + dataset
    #load
    if dataset == '50salad':
        load_base = '/datasets/50salads/hyperbolic_embedding/mapping_json/'
    elif dataset == 'breakfast':
        load_base = '/datasets/breakfast/hyperbolic_embedding/mapping_json/'
    else:
        raise "Not implemented"
    
    with open(load_base + '/hypernyms_reverse.json', 'r') as f:
        hypernym = json.load(f)
    with open(load_base +'/actions_reverse.json', 'r') as f:
        action = json.load(f)
    
    #load 이유는 hyp, action 경계 알기 위해서임 COSNE embedding은 [hyp, action] 순으로 stack 되어있음.
    hypernym_num = len(hypernym.keys())
    hypernym = embeddings[:hypernym_num, :]
    action = embeddings[hypernym_num :, :]
    #save embeddings. cosne embedding is already np array
    np.save(save_emb_path + '/hypernym_{}_poincare_{}D_.npy'.format(dataset, embeddings.shape[1]), hypernym)
    np.save(save_emb_path + '/action_{}_poincare_{}D_.npy'.format(dataset, embeddings.shape[1]), action)


#만약 plot하려면 action 목록까지 이미 가지고 있어야함. 데이터 셋 어떤 것 쓰는지 알아야함.. 따로 빼자
def plot_COSNE(embeddings, dataset):
    if embeddings.shape[1] != 2:
        raise Exception("Embedding is not 2D. Cannot visualize")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)

    #scatter embeddings
    plt.scatter(embeddings[:,0], embeddings[:,1], s=30)

    # Annotation
    if dataset == '50salad':
        #salad whole annotation
        acts = ['cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad', 'peel_cucumber', 'cut_cucumber', 'place_cucumber_into_bowl', 'cut_tomato', 'place_tomato_into_bowl', 'cut_cheese', 'place_cheese_into_bowl', 'cut_lettuce', 'place_lettuce_into_bowl', 'mix_ingredients', 'add_oil', 'add_vinegar', 'add_salt', 'add_pepper', 'mix_dressing', 'serve_salad_onto_plate', 'add_dressing']
    elif dataset == 'breakfast':
        #only breakfast hypernym
        acts = ['pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']
    elif dataset == 'composition':
        #salad whole annotation + breakfast hypernym
        acts = ['cut_and_mix_ingredients', 'prepare_dressing', 'serve_salad', 'peel_cucumber', 'cut_cucumber', 'place_cucumber_into_bowl', 'cut_tomato', 'place_tomato_into_bowl', 'cut_cheese', 'place_cheese_into_bowl', 'cut_lettuce', 'place_lettuce_into_bowl', 'mix_ingredients', 'add_oil', 'add_vinegar', 'add_salt', 'add_pepper', 'mix_dressing', 'serve_salad_onto_plate', 'add_dressing', 'pour_cereals', 'take_bowl', 'stir_cereals', 'pour_milk', 'pour_oil', 'stirfry_egg', 'put_egg2plate', 'add_saltnpepper', 'crack_egg', 'take_plate', 'take_eggs', 'butter_pan', 'pour_egg2pan', 'stir_egg', 'stir_dough', 'fry_pancake', 'pour_dough2pan', 'spoon_flour', 'put_pancake2plate', 'pour_flour', 'put_bunTogether', 'take_knife', 'smear_butter', 'take_butter', 'take_topping', 'cut_bun', 'put_toppingOnTop', 'peel_fruit', 'put_fruit2bowl', 'stir_fruit', 'cut_fruit', 'pour_water', 'add_teabag', 'spoon_sugar', 'pour_sugar', 'stir_tea', 'take_cup', 'take_glass', 'pour_juice', 'cut_orange', 'take_squeezer', 'squeeze_orange', 'stir_milk', 'spoon_powder', 'fry_egg', 'stir_coffee', 'pour_coffee']

    for i, txt in enumerate(acts):
        plt.annotate(txt, (embeddings[i,0], embeddings[i,1]))
    ax.set_aspect('equal')
    plt.axis('off')

    plt.savefig("./saved_figures/{}" + "CO-SNE.png".format(dataset), bbox_inches='tight', dpi=fig.dpi)
    


if __name__ == "__main__":
    """
    embeddings, colors = generate_high_dims()


    learning_rate = 5.0
    learning_rate_for_h_loss = 0.1
    perplexity = 20
    early_exaggeration = 1.0
    student_t_gamma = 0.1


    tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding  = run_TSNE(embeddings, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)

    plot_low_dims(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
    """
    # load_embeddings('50salad', 512)
    dataset = 'breakfast'
    dim=50
    n_components = 2
    n_iter = 510
    #단일 break은 520, 510이 좋음, salad은 1000. 500까지는 거의 한 점 수준으로 모임 이후부터 퍼짐  
    #혼합은 520

    learning_rate = 1.0
    learning_rate_for_h_loss = 0.1
    perplexity = 15
    early_exaggeration = 1.0
    student_t_gamma = 0.1


    # 10차원으로 만든 브렉을 모델에 태우는 것 까지..
    #for composition
    # embeddings = generate_compositional(50)
    embeddings = generate_embeddings(dataset, dim)
    CO_SNE_embedding = run_COSNE(embeddings, n_components, learning_rate , learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma, n_iter)
    save_COSNE(CO_SNE_embedding, dataset)
    if n_components == 2: 
        #plot and save figure
        plot_COSNE(CO_SNE_embedding, dataset)


    
