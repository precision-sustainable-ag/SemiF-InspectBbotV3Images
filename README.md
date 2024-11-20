# SemiF-InspectBbotV3Images

### **batch_collection_plot.py**
This script generates bar plots to visualize the number of JPG and RAW images collected across different batches, categorized by collection sites and dates. It reads batch information from a specified file, calculates image counts for each batch, and organizes the data by site for plotting. The resulting plots highlight temporal trends in image collection, aiding in identifying data gaps or inconsistencies, and are saved in an output directory for reporting purposes.

### **inspect_images.py**
This script inspects batches of images to assess their capture timing and quality. It processes images based on sampling strategies (e.g., random, first, last, or middle), resizes them for visualization, and extracts metadata like exposure and ISO. Plots are created to show capture times, highlighting any irregularities in timing or image metadata. The tool is highly configurable, supporting batch-level inspection and detailed reporting on image attributes.

### **storage_report.py**
This script generates a comprehensive storage report for batches of images. It calculates and logs statistics such as the number of JPG and RAW files, their respective sizes in MiB, and the total folder size in GiB. Additionally, it aggregates storage usage across all batches and reports the total size in TiB. The output is a detailed text report, which serves as an essential tool for monitoring storage utilization and planning resource allocation.