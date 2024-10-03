#!/usr/bin/env python
# coding: utf-8

# In[16]:
import numpy as np
import matplotlib.pyplot as plt
from microboone_utils import *
import pandas as pd
from h5py import File

import argparse
from skimage.measure import block_reduce
from math import floor, ceil
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gerador de imagens', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename',default='./bnb_WithWire_00.h5', type=str, help='Specify the file name(.h5)')

    args = parser.parse_args()
    filename=args.filename
    # In[17]:
    #abre o arquivo
    f = File(filename,'r')
    filename=os.path.splitext(filename)[0]
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

    parser.add_argument('-n', '--n_events',default=len(event_table_pd), type=int, help='Specify the the number of events that you want')
    args = parser.parse_args()

    # In[20]:
    #montar a tabela de wire
    N_fotos=args.n_events
    end_hit=0
    end_edep=0

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
        plt.savefig(f"./figuras/"+filename+f"_event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_0.png",  bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a segunda imagem
        plt.imshow(planeadcs[1].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
        plt.axis('off')
        # Salva a segunda imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/"+filename+f"_event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_1.png", bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a terceira imagem
        plt.imshow(planeadcs[2].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
        plt.axis('off')
        # Salva a terceira imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/"+filename+f"_event_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_2.png", bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        #mapa de classificao agora -----
                #vamos mexer na hit table id

        hit_table=f[labels[2]]
        event_hit_table=hit_table["event_id"]
        names_event=["run","subrun","event"]

        #buscar o inicio do evento e o final.
        init_hit=end_hit #lembra depois no for de mudar os paramentros iniciais, para ser mais rapido
        found_flag=False

        #procurando os indices
        for aux_hit,values in enumerate(event_hit_table[init_hit::]):
            aux_hit=aux_hit+init_hit
            if values[0]==this_run and values[1]==this_subrun and values[2]==this_event and found_flag==False:
                init_hit=aux_hit
                found_flag=True
            if (values[0]!=this_run or values[1]!=this_subrun or values[2]!=this_event) and found_flag==True:
                end_hit=aux_hit
                found_flag=False
                break
        #----
        names_event_values=[[] for _ in range(len(names_event))]
        for i in range(end_hit-init_hit):
            for j in range(len(names_event)):
                names_event_values[j].append(event_hit_table[i+init_hit][j])

        temp_dict = {}
        for i, name in enumerate(names_event):
            temp_dict[name] = names_event_values[i]

        hit_id=hit_table["hit_id"][init_hit:end_hit]
        local_plane=hit_table["local_plane"][init_hit:end_hit]
        local_time=hit_table["local_time"][init_hit:end_hit]
        local_wire=hit_table["local_wire"][init_hit:end_hit]
        rms=hit_table["rms"][init_hit:end_hit]

        temp_dict["hit_id"]=hit_id.flatten()
        hit_id=None
        temp_dict["local_plane"]=local_plane.flatten()
        local_plane=None
        temp_dict["local_time"]=local_time.flatten()
        local_time=None
        temp_dict["local_wire"]=local_wire.flatten()
        local_wire=None
        temp_dict["rms"]=rms.flatten()
        rms=None

        hit_table=None
        event_hit_table=None

        hit_table_pd = pd.DataFrame(temp_dict)

        #vamos mexer na edep table id
        edep_table=f[labels[0]]
        event_edep_table=edep_table["event_id"]
        names_event=["run","subrun","event"]

        #buscar o inicio do evento e o final.
        init_edep=end_edep#lembra depois no for de mudar os paramentros iniciais, para ser mais rapido
    
        found_flag=False

        #procurando os indices # colocar logica de nao achar o proximo
        for aux_edep,values in enumerate(event_edep_table[init_edep::]):
            aux_edep=aux_edep+init_edep
            if values[0]==this_run and values[1]==this_subrun and values[2]==this_event and found_flag==False:
                init_edep=aux_edep
                found_flag=True
            if (values[0]!=this_run or values[1]!=this_subrun or values[2]!=this_event) and found_flag==False: #esse aqui eh caso o proximo nao exista
                end_edep=aux_edep
                found_flag=False
                break
            if (values[0]!=this_run or values[1]!=this_subrun or values[2]!=this_event) and found_flag==True:
                end_edep=aux_edep
                found_flag=False
                break

        #----
        names_event_values=[[] for _ in range(len(names_event))]
        for i in range(end_edep-init_edep):
            for j in range(len(names_event)):
                names_event_values[j].append(event_edep_table[i+init_edep][j])

        temp_dict = {}
        for i, name in enumerate(names_event):
            temp_dict[name] = names_event_values[i]

        energy_fraction=edep_table["energy_fraction"][init_edep:end_edep]
        hit_id=edep_table["hit_id"][init_edep:end_edep]
        g4_id=edep_table["g4_id"][init_edep:end_edep]

        temp_dict["energy_fraction"]=energy_fraction.flatten()
        energy_fraction=None
        temp_dict["hit_id"]=hit_id.flatten()
        hit_id=None
        temp_dict["g4_id"]=g4_id.flatten()
        g4_id=None

        edep_table=None
        event_edep_table=None

        edep_table_pd = pd.DataFrame(temp_dict)

        edep_table_pd = edep_table_pd.sort_values(by=['energy_fraction'], ascending=False, kind='mergesort').drop_duplicates(["hit_id"])
        hit_table_pd = hit_table_pd.merge(edep_table_pd, on=["hit_id"], how="left")
        hit_table_pd['g4_id'] = hit_table_pd['g4_id'].fillna(-1)
        hit_table_pd = hit_table_pd.fillna(0)

        
        planetruth = [np.zeros(shape=(nwires(p),ntimeticks())) for p in range(0,nplanes())]
        nrms = 2
        for p in range(0,nplanes()):
            nuhits = hit_table_pd.query('local_plane==%i and g4_id>=0'%p)[['local_wire','local_time','rms']]
            for i,h in nuhits.iterrows():
                planetruth[p][int(h['local_wire'])][floor(h['local_time']-nrms*h['rms']):ceil(h['local_time']+nrms*h['rms'])] = 1
            cosmhits = hit_table_pd.query('local_plane==%i and g4_id<0'%p)[['local_wire','local_time','rms']]
            for i,h in cosmhits.iterrows():
                planetruth[p][int(h['local_wire'])][floor(h['local_time']-nrms*h['rms']):ceil(h['local_time']+nrms*h['rms'])] = -1

        for p in range(0,nplanes()):
            planetruth[p] = block_reduce(planetruth[p], block_size=(1,f_downsample), func=np.sum)


        # Plota a primeira imagem
        plt.imshow(planetruth[0].T, vmin=-1, vmax=1, origin='lower', cmap='coolwarm')
        plt.axis('off')
        # Salva a primeira imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/"+filename+f"_hit_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_0.png",  bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a segunda imagem
        plt.imshow(planetruth[1].T, vmin=-1, vmax=1, origin='lower', cmap='coolwarm')
        plt.axis('off')
        # Salva a segunda imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/"+filename+f"_hit_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_1.png",  bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem

        # Plota a terceira imagem
        plt.imshow(planetruth[2].T, vmin=-1, vmax=1, origin='lower', cmap='coolwarm')
        plt.axis('off')
        # Salva a terceira imagem
        plt.tight_layout()
        plt.savefig(f"./figuras/"+filename+f"_hit_{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_plane_2.png",  bbox_inches='tight', pad_inches=0)
        plt.clf()  # Limpa a figura para a próxima imagem
                
