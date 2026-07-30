from fileScan import scanFiles

results = scanFiles(
    folderData=r"C:/git/Trieste/Data",
    sampleName="FeRh_A04",
    scanNames=["Scan", "", "NoProbe", "OnlyProbe", "Dark"],
    scanNo=68,
    verbose=True,
    minimumFileSizeRatio=0.95,
)
