import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects, dilation
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
import pandas as pd
from io import BytesIO
import zipfile

st.title("Nuclei Segmentation and RFP Analysis")

def reset_app():
    keys_to_clear = [
        "analysis_done",
        "c1_img",
        "c2_img",
        "nuclei_mask",
        "labeled_nuclei",
        "results",
        "file_name",
        "c1_file",
        "c2_file",
        "file_name_input"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.info("Results cleared. You can upload new files and run analysis again.")

# File uploaders and text input with keys
c1_file = st.file_uploader("Upload C1 image (Nup96-mEGFP)", type=["tif", "tiff"], key="c1_file")
c2_file = st.file_uploader("Upload C2 image (RFP)", type=["tif", "tiff"], key="c2_file")
file_name = st.text_input("Enter output file name", key="file_name_input")

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if st.button("Run Analysis") or st.session_state.analysis_done:
    if not c1_file or not c2_file or not file_name:
        st.error("Please make sure both images are uploaded and a file name is entered.")
        st.stop()

    if not st.session_state.analysis_done:
        st.success("Files successfully loaded!")
    else:
        st.info("Analysis results loaded from memory.")

    if not st.session_state.analysis_done:
        st.session_state.c1_img = tiff.imread(c1_file)
        st.session_state.c2_img = tiff.imread(c2_file)

        st.info("Beginning segmentation...")
        blurred = gaussian(st.session_state.c1_img, sigma=3)
        thresh = threshold_otsu(blurred.flatten())
        nuclei_mask = blurred > thresh
        nuclei_mask = binary_closing(nuclei_mask, disk(1))
        nuclei_mask = remove_small_objects(nuclei_mask, min_size=1000)
        nuclei_mask = binary_fill_holes(nuclei_mask)
        labeled_nuclei = label(nuclei_mask)

        st.session_state.nuclei_mask = nuclei_mask
        st.session_state.labeled_nuclei = labeled_nuclei

        st.write(f"Total nuclei found: {len(np.unique(labeled_nuclei))}")
        st.success("Segmentation complete!")

        st.info("Beginning analysis of nuclei...")
        results = []
        num_nuclei = len(np.unique(labeled_nuclei)) - 1
        count = 0
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for region_label in np.unique(labeled_nuclei):
            if region_label == 0:
                continue
            nucleus_mask = labeled_nuclei == region_label
            ys, xs = np.where(nucleus_mask)
            touches_border = ys.min() == 0 or ys.max() == st.session_state.c1_img.shape[0]-1 or xs.min() == 0 or xs.max() == st.session_state.c1_img.shape[1]-1
            if touches_border:
                results.append({
                    "NucleusLabel": region_label,
                    "NucleusMeanRFP": np.nan,
                    "CytoplasmMeanRFP": np.nan,
                    "NucleusMeanRFP/CytoplasmMeanRFP": np.nan,
                    "IncompleteMask": True
                })
                count += 1
                progress_bar.progress(count / max(num_nuclei,1))
                progress_text.text(f"Nuclei processed: {count}/{num_nuclei}")
                continue

            outer_ring, inner_ring = dilation(nucleus_mask, disk(8)), dilation(nucleus_mask, disk(2))
            cytoplasm_mask = outer_ring & ~inner_ring
            nucleus_intensity = st.session_state.c2_img[nucleus_mask].mean()
            cytoplasm_intensity = st.session_state.c2_img[cytoplasm_mask].mean()
            results.append({
                "NucleusLabel": region_label,
                "NucleusMeanRFP": nucleus_intensity,
                "CytoplasmMeanRFP": cytoplasm_intensity,
                "NucleusMeanRFP/CytoplasmMeanRFP": nucleus_intensity/cytoplasm_intensity
            })

            count += 1
            progress_bar.progress(count / max(num_nuclei,1))
            progress_text.text(f"Nuclei processed: {count}/{num_nuclei}")

        st.session_state.results = results
        st.session_state.file_name = file_name
        st.session_state.analysis_done = True

    df = pd.DataFrame(st.session_state.results)
    st.write(df)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(st.session_state.c2_img, cmap="Reds", alpha=0.6)
    ax.imshow(st.session_state.nuclei_mask, cmap="nipy_spectral", alpha=0.4)
    ax.axis("off")
    ax.set_title("Overlay: Nuclei and Mask")
    props = regionprops(st.session_state.labeled_nuclei)
    for prop in props:
        y,x = prop.centroid
        ax.text(x,y,str(prop.label),color="white",fontsize=8,ha="center",va="center",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.6))
    st.pyplot(fig)

    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr(f"intensity_results_{file_name}.csv", csv_buffer.getvalue())
        zf.writestr(f"overlay_{file_name}.png", img_buffer.getvalue())
    zip_buffer.seek(0)

    st.download_button(
        label="Download CSV + Overlay Image (ZIP)",
        data=zip_buffer,
        file_name=f"{file_name}_results.zip",
        mime="application/zip"
    )

    st.markdown("---")
    st.button("Reset Inputs", on_click=reset_app)
