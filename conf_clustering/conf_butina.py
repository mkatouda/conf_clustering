#!/usr/bin/env python

import os
import sys
import argparse

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina

def get_parser():
    class customHelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                              argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=customHelpFormatter,
        description="Butina clustering code of conformers"
    )
    parser.add_argument(
        '-s', '--sdf', type=str,
        help = 'input conformation sdf file'
    )
    parser.add_argument(
        '--cutoff-rms', type=float, default=2.0,
        help = 'cutoff of Root Mean Square (RMS) deviation clustering'
    )
    parser.add_argument(
        '--cutoff-tfd', type=float, default=0.04,
        help = 'cutoff of Torsion Fingerprint Deviation (TFD) clustering'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', dest='verbose',
        help='Print verbose output'
    )
    args = parser.parse_args()
    return args 

def get_cluster_center(clusters):
    return [c[0] for c in clusters]

def plot_cluster_profile(clusters, pngfile):
    fig = plt.figure(1, figsize=(10, 8))
    plt1 = plt.subplot(111)
    plt.axis([0, len(clusters), 0, len(clusters[0])+1])
    plt.xlabel('Cluster index', fontsize=20)
    plt.ylabel('Number of molecules', fontsize=20)
    plt.tick_params(labelsize=16)
    plt1.bar(range(1, len(clusters)), [len(c) for c in clusters[:len(clusters)-1]], lw=0)
    plt.savefig(pngfile)
    plt.clf()
    plt.close()

def conf_butina_main(sdfin_path, cutoff_rms=2.0, cutoff_tfd=0.3, verbose=False):
    mols = list(Chem.SDMolSupplier(sdfin_path, removeHs=True))
    print('mols:', len(mols))
    #mols = mols[0:100] # for ebug

    mol = Chem.Mol(mols[0])
    mol.RemoveAllConformers()
    for m in mols:
        #print(Chem.MolToMolBlock(m))
        mol.AddConformer(m.GetConformer(), assignId=True)
    print('SMILES:', Chem.MolToSmiles(mol), '\nNumber of conformations:', mol.GetNumConformers())

    rms_mat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
    print('rms_mat:', type(rms_mat), len(rms_mat)) #, '\n', rms_mat)

    tfd_mat = TorsionFingerprints.GetTFDMatrix(mol)

    num = mol.GetNumConformers()
    rms_clusters = Butina.ClusterData(rms_mat, num, cutoff_rms, isDistData=True, reordering=True)
    tfd_clusters = Butina.ClusterData(tfd_mat, num, cutoff_tfd, isDistData=True, reordering=True)
    print('Number of RMS clusters:', len(rms_clusters))
    if verbose:
        for i in range(len(rms_clusters)):
            print('RMS cluster_{}:'.format(i), len(rms_clusters[i]), rms_clusters[i])
    print('Number of TFD clusters:', len(tfd_clusters))
    if verbose:
        for i in range(len(tfd_clusters)):
            print('TFD cluster_{}:'.format(i), len(tfd_clusters[i]), tfd_clusters[i])

    mols = list(Chem.SDMolSupplier(sdfin_path, removeHs=False))
    sdfout_path = os.path.splitext(sdfin_path)[0] + '_rms_clusters_center.sdf'
    writer = Chem.SDWriter(sdfout_path)
    for c in rms_clusters:
        writer.write(mols[c[0]])
    writer.close()

    sdfout_path = os.path.splitext(sdfin_path)[0] + '_tfd_clusters_center.sdf'
    writer = Chem.SDWriter(sdfout_path)
    for c in tfd_clusters:
        writer.write(mols[c[0]])
    writer.close()

    pngout_path = os.path.splitext(sdfin_path)[0] + '_rms_clusters_profile.png'
    plot_cluster_profile(rms_clusters, pngout_path)
    pngout_path = os.path.splitext(sdfin_path)[0] + '_tfd_clusters_profile.png'
    plot_cluster_profile(tfd_clusters, pngout_path)

def main():
    args = get_parser()
    if args.verbose: print(args)

    sdfin_path = args.sdf
    cutoff_rms = args.cutoff_rms
    cutoff_tfd = args.cutoff_tfd
    verbose = args.verbose

    conf_butina_main(sdfin_path, cutoff_rms=cutoff_rms, cutoff_tfd=cutoff_tfd,
                     verbose=verbose)

if __name__ == '__main__':
    main()
