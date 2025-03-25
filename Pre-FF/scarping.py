#!/usr/bin/env python3
"""
Fixed RNA Regulatory Elements Collector
Retrieves RNA binding proteins, UTRs, and miRNAs from public databases
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from Bio import Entrez, SeqIO

Entrez.email = "rishabh.webdev@gmail.com"

# Search terms for different RNA regulatory elements
SEARCH_TERMS = {
    'RBP': "RNA binding[Molecular Function] AND Homo sapiens[Organism] AND alive[Property]",
    'UTR': "(UTR binding[Molecular Function] OR UTR[All Fields]) AND Homo sapiens[Organism] AND alive[Property]",
    'miRNA': "microRNA[Gene Type] AND Homo sapiens[Organism] AND alive[Property]"
}

# Limits for each category
LIMITS = {
    'RBP': 30,
    'UTR': 20,
    'miRNA': 30
}

def fetch_genes_from_ncbi(search_term, element_type, limit):
    """Fetch genes from NCBI based on search term"""
    print(f"Fetching {element_type} elements...")
    
    # Search for genes
    handle = Entrez.esearch(db="gene", term=search_term, retmax=limit)
    record = Entrez.read(handle)
    handle.close()
    
    gene_ids = record["IdList"]
    print(f"  Found {len(gene_ids)} gene IDs for {element_type}")
    
    elements = []
    
    # Get details for each gene
    for i, gene_id in enumerate(gene_ids):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(gene_ids)}")
            
        try:
            # Use summary instead of fetch - different response format
            handle = Entrez.esummary(db="gene", id=gene_id)
            summary = Entrez.read(handle)
            handle.close()
            
            # Extract gene info - handle DocumentSummarySet structure
            if 'DocumentSummarySet' in summary:
                doc_summary = summary['DocumentSummarySet']['DocumentSummary'][0]
                symbol = doc_summary.get('Name', '')
                name = doc_summary.get('Description', '')
                
                if symbol:
                    element = {
                        'name': name,
                        'symbol': symbol,
                        'type': element_type,
                        'gene_id': gene_id,
                        'source': 'NCBI'
                    }
                    elements.append(element)
                    print(f"    Added element: {symbol}")
            
            time.sleep(0.5)  # Be nice to NCBI servers
            
        except Exception as e:
            print(f"  Error fetching gene {gene_id}: {e}")
    
    return elements

def fetch_genes_direct(element_type):
    """Fetch gene information using a predefined list of symbols"""
    print(f"Using direct method for {element_type} genes")
    elements = []
    
    # Known genes for each type
    known_genes = {
        'RBP': ['ELAVL1', 'TIA1', 'HNRNPA1', 'HNRNPC', 'HNRNPD', 'PTBP1', 'IGF2BP1', 'SRSF1', 
                'FMR1', 'CPEB1', 'ZFP36', 'QKI', 'KHDRBS1', 'RBFOX1', 'MBNL1'],
                
        'UTR': ['SECISBP2', 'ACO1', 'ELAVL1', 'DHX36', 'CPEB1', 'CPSF1', 'EIF1', 'PABPC1',
                'YTHDF1', 'YTHDF2', 'METTL3', 'METTL14', 'PCBP2'],
                
        'miRNA': ['MIR155', 'MIR146A', 'MIR17', 'MIR21', 'MIR150', 'MIR181A1', 'MIR142', 
                 'MIR223', 'MIR34A', 'MIR200C', 'MIR122', 'MIR375', 'LET7A1']
    }
    
    if element_type not in known_genes:
        print(f"No known genes for {element_type}")
        return []
    
    genes = known_genes[element_type]
    
    for i, symbol in enumerate(genes):
        print(f"  Processing {i+1}/{len(genes)}: {symbol}")
        
        try:
            # Search for the gene
            handle = Entrez.esearch(db="gene", term=f"{symbol}[Gene Symbol] AND human[Organism]")
            record = Entrez.read(handle)
            handle.close()
            
            if record["Count"] == "0":
                print(f"  No gene found for {symbol}")
                continue
                
            # Get gene ID
            gene_id = record["IdList"][0]
            
            # Get gene details
            handle = Entrez.esummary(db="gene", id=gene_id)
            summary = Entrez.read(handle)
            handle.close()
            
            # Get gene name
            name = ""
            if 'DocumentSummarySet' in summary:
                doc_summary = summary['DocumentSummarySet']['DocumentSummary'][0]
                name = doc_summary.get('Description', '')
            
            element = {
                'name': name,
                'symbol': symbol,
                'type': element_type,
                'gene_id': gene_id,
                'source': 'Direct'
            }
            elements.append(element)
            print(f"    Added: {symbol}")
            
            time.sleep(0.5)  # Be nice to NCBI servers
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
    
    return elements

def get_sequence_data(gene_symbol):
    """Get mRNA sequence data from NCBI for a gene symbol"""
    try:
        # Search for the gene
        handle = Entrez.esearch(db="gene", term=f"{gene_symbol}[Gene Symbol] AND human[Organism]")
        record = Entrez.read(handle)
        handle.close()
        
        if record["Count"] == "0":
            print(f"No gene found for {gene_symbol}")
            return None
            
        # Get gene ID
        gene_id = record["IdList"][0]
        
        # Init result dictionary
        result = {'gene_id': gene_id, 'symbol': gene_symbol}
        
        # Search for mRNA sequences
        handle = Entrez.esearch(
            db="nuccore", 
            term=f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism] AND mRNA[Filter] AND refseq[Filter]"
        )
        nuccore_record = Entrez.read(handle)
        handle.close()
        
        # If no direct mRNA found, try to find linked sequences
        if nuccore_record["Count"] == "0":
            handle = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id)
            link_results = Entrez.read(handle)
            handle.close()
            
            nucleotide_ids = []
            for linksetdb in link_results[0].get("LinkSetDb", []):
                if linksetdb["DbTo"] == "nuccore":
                    for link in linksetdb["Link"]:
                        nucleotide_ids.append(link["Id"])
                        
            if not nucleotide_ids:
                return result
                
            # Use first ID
            mrna_id = nucleotide_ids[0]
        else:
            # Use direct mRNA hit
            mrna_id = nuccore_record["IdList"][0]
        
        # Get sequence data
        handle = Entrez.efetch(db="nuccore", id=mrna_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        # Extract sequences
        result['sequence'] = str(record.seq)
        result['accession'] = record.id
        
        # Find coding sequence and UTRs
        for feature in record.features:
            if feature.type == "CDS":
                result['cds_sequence'] = str(feature.extract(record.seq))
                if 'protein_id' in feature.qualifiers:
                    result['protein_id'] = feature.qualifiers['protein_id'][0]
            elif feature.type == "5'UTR":
                result['five_utr'] = str(feature.extract(record.seq))
            elif feature.type == "3'UTR":
                result['three_utr'] = str(feature.extract(record.seq))
        
        return result
        
    except Exception as e:
        print(f"Error processing gene {gene_symbol}: {str(e)}")
        return None

def main():
    """Main function to collect RNA regulatory data"""
    # Collect genes for each element type
    all_elements = []
    
    for element_type, search_term in SEARCH_TERMS.items():
        # Try the API search first
        elements = fetch_genes_from_ncbi(search_term, element_type, LIMITS[element_type])
        
        # If no elements found, use direct method with known genes
        if len(elements) == 0:
            print(f"No {element_type} elements found via API search, trying direct method")
            elements = fetch_genes_direct(element_type)
        
        all_elements.extend(elements)
        print(f"Found {len(elements)} {element_type} elements")
    
    # Get sequence data for each element
    results = []
    for i, element in enumerate(all_elements):
        print(f"Processing {i+1}/{len(all_elements)}: {element['symbol']}")
        
        # Get sequence data
        seq_data = get_sequence_data(element['symbol'])
        
        if not seq_data:
            print(f"  No sequence data found for {element['symbol']}")
            continue
        
        # Combine element info with sequence data
        combined = {**element, **seq_data}
        results.append(combined)
        
        # Be nice to the API
        time.sleep(1)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("rna_elements.csv", index=False)
    print(f"Saved {len(results)} records to rna_elements.csv")
    
    # Create NumPy arrays for model training
    # if len(results) > 0:
    #     # Filter for valid CDS sequences
    #     valid_df = df[df['cds_sequence'].notna() & (df['cds_sequence'] != '')]
        
    #     if len(valid_df) > 0:
    #         # Save sequences and types
    #         np.save("all_sequences.npy", valid_df['cds_sequence'].values)
    #         np.save("all_types.npy", valid_df['type'].values)
    #         np.save("all_symbols.npy", valid_df['symbol'].values)
            
    #         print(f"Saved {len(valid_df)} sequences for CodonBERT fine-tuning")
            
    #         # Save type-specific files
    #         for element_type in valid_df['type'].unique():
    #             type_df = valid_df[valid_df['type'] == element_type]
    #             if len(type_df) > 0:
    #                 np.save(f"{element_type}_sequences.npy", type_df['cds_sequence'].values)
    #                 np.save(f"{element_type}_symbols.npy", type_df['symbol'].values)
    #                 print(f"Saved {len(type_df)} {element_type} sequences")

if __name__ == "__main__":
    main()