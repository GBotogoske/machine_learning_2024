#!/usr/bin/env python
# coding: utf-8

# In[16]:
import numpy as np
import matplotlib.pyplot as plt
from microboone_utils import *
import pandas as pd
from h5py import File

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gerador de imagens', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename',default='/home/gabriel/Downloads/bnb_WithWire_00.h5', type=str, help='Specify the file name(.h5)')
    parser.add_argument('-n', '--n_events',default=1, type=int, help='Specify the the number of events that you want')

    args = parser.parse_args()
    # In[17]:
    #abre o arquivo
    f = File(args.filename,'r')

    # In[18]:
    #mostra todas as tabelas disponiveis no arquivo .h5 aberto
    labels=[]
    sub_labels=[]
    for lines in f:
            labels.append(lines)
            aux=[]
            for key in f[lines]:
                aux.append(key)
            sub_labels.append(aux)


    # In[19]:


    #montar uma tbela dos eventos disponiveis na file
    size=len(f[labels[1]]["event_id"][:])

    names_event=["run","subrun","event"]
    names_event_values=[[] for _ in range(len(names_event))]
    for i in range(size):
        for j in range(len(names_event)):
            names_event_values[j].append(f[labels[1]]["event_id"][i][j])

    temp_dict = {}
    for i, name in enumerate(names_event):
        temp_dict[name] = names_event_values[i]

    
    event_table_pd = pd.DataFrame(temp_dict)
    names_event_values.clear()    


    # In[20]:
    #montar a tabela de wire
    N_fotos=args.n_events
    for evento_index in range(N_fotos):
        max=8256
        init=max*evento_index
        end=max*(evento_index+1) #tentar ver depois como pegar evento por evento melhor!!!! acabei de ver que sao 8256 linhas por evento

        wire_table=f[labels[10]]
        event_wire_plan=wire_table["event_id"][init:end]
        local_plane = wire_table["local_plane"][init:end]
        adc=wire_table["adc"][init:end]
        adc = [adc.tolist() for adc in adc]

        names_event=["run","subrun","event"]
        names_event_values=[[] for _ in range(len(names_event))]
        for i in range(max):
            for j in range(len(names_event)):
                names_event_values[j].append(wire_table["event_id"][i+init][j])

        temp_dict = {}
        for i, name in enumerate(names_event):
            temp_dict[name] = names_event_values[i]

        temp_dict["local_plane"]=local_plane.flatten()
        temp_dict["adc"]=adc


        wire_table_pd = pd.DataFrame(temp_dict)

        adc.clear()
        event_wire_plan=None
        local_plane=None
        wire_table=None


        # In[21]:


        #vamos filtar por eventos, depois esse bloco tem que ser um for, junto com a parte de cima ... uhmmm
        my_index=evento_index
        this_run=event_table_pd["run"][my_index]
        this_subrun=event_table_pd["subrun"][my_index]
        this_event=event_table_pd["event"][my_index]

        wire_table_pd=wire_table_pd.query("run==@this_run and subrun==@this_subrun and event==@this_event")


        # In[22]:


        planeadcs=[]
        for p in range(0,nplanes()):
            table_aux=wire_table_pd.query("local_plane==@p")
            aux=[]
            for j in range(len(table_aux)):
                aux.append(table_aux.iloc(0)[j]["adc"])
            planeadcs.append(np.array(aux))

        wire_table_pd=wire_table_pd.drop(columns="adc")


        # In[23]:


        from skimage.measure import block_reduce
        f_downsample = 6
        for p in range(0,nplanes()):
            planeadcs[p] = block_reduce(planeadcs[p], block_size=(1,f_downsample), func=np.sum)

        adccutoff = 10.*f_downsample/6.
        adcsaturation = 100.*f_downsample/6.
        for p in range(0,nplanes()):
            planeadcs[p][planeadcs[p]<adccutoff] = 0
            planeadcs[p][planeadcs[p]>adcsaturation] = adcsaturation


        # In[29]:
        evt_id = [this_run, this_subrun, this_event]
        zmax = adcsaturation

        print("Run / Sub / Event : %i / %i / %i - saturation set to ADC sum=%.2f" % (evt_id[0], evt_id[1], evt_id[2], zmax))

        # Plota a primeira imagem
        plt.imshow(planeadcs[0].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
        plt.axis('off')

        # Salva a primeira imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_0.png",  bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a segunda imagem
        plt.imshow(planeadcs[1].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
        plt.axis('off')
        # Salva a segunda imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_1.png", bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a terceira imagem
        plt.imshow(planeadcs[2].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
        plt.axis('off')
        # Salva a terceira imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_2.png", bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

