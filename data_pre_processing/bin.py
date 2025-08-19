import pandas as pd

#read the file
df = pd.read_csv('/public_data/liukaihua/GM12878_FC5/GSM6284586_GM12878_FC5_reads_alignment.csv')
result = {}
#filter
for _, row in df.iterrows():
   
    if (
        float(row['FragMatchratio']) >= 0.8 and
        int(row['MapQual']) >= 30 and
        int(row['Matches']) >= 100 and
        row['LRvdF_pfix'] == True and
        int(row['LRvdF_pdist']) <= 1000
        # row['Note'] == "FirstFilter"
    ):

        read_name = row["read_name"]
        bin = int((int(row['start']) + int(row['end'])) / 2) + 1
        bin = f"{row['chrom']}:{bin}"
        if read_name not in result:
            result[read_name] = []
        result[read_name].append(bin)


output_file = "HiPore-C_filt.clusters"  

#save
with open(output_file, "w") as f:
    for read_name, positions in result.items():
        bins_str = " ".join(positions) 
        f.write(f"{read_name} {bins_str}\n") 

print(f"The results have been saved to {output_file}")