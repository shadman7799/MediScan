const openCameraBtn = document.getElementById("openCameraBtn");
const webcam = document.getElementById("webcam");
const webcamContainer = document.getElementById("webcam-container");

const cropBtn = document.getElementById("btn-crop");

const captureBtn = document.getElementById("captureBtn");
const croppedOutput = document.getElementById('outputImg');
const croppedContainer = document.getElementById('cropped-container')
const capturedImage = document.getElementById("capturedImage");
const capturedImageContainer = document.getElementById(
  "capturedImageContainer"
);
captureBtn.classList.add("hidden");
const submitFinalBtn = document.getElementById('submitFinalBtn');
let stream;

openCameraBtn.addEventListener("click", async function () {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    webcamContainer.style.display = "block";
    captureBtn.style.display = "block";
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
});

openCameraBtn.addEventListener("click", async function () {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    webcamContainer.style.display = "block";
  } catch (error) {
    console.error("Error accessing webcam:", error);
  }
});

const fileInput = document.getElementById("fileInput");

captureBtn.addEventListener("click", function () {
  if (stream) {
    // Create a canvas to draw the current frame from the video element
    const canvas = document.createElement("canvas");
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    const context = canvas.getContext("2d");

    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);

    // Convert the canvas content to a data URL (base64-encoded image)
    const imageDataUrl = canvas.toDataURL("image/png");

    // Display the captured image
    capturedImage.src = imageDataUrl;
    // capturedImageContainer.style.display = "block";


    // Hide the webcam section
    webcamContainer.style.display = "none";
    captureBtn.style.display = "none";


    // Set the captured image as the value for the file input
    const blob = dataURLtoBlob(imageDataUrl);
    const file = new File([blob], "captured_image.png");
    const fileList = new DataTransfer();
    fileList.items.add(file);
    fileInput.files = fileList.files;
    performCrop();
  }
});
function dataURLtoBlob(dataURL) {
  const byteString = atob(dataURL.split(",")[1]);
  const mimeString = dataURL.split(",")[0].split(":")[1].split(";")[0];
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);

  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }

  return new Blob([arrayBuffer], { type: mimeString });
}
// Trigger input file click when the captured image is clicked
capturedImage.addEventListener("click", function () {
  fileInput.click();
});

const imageContainer = document.querySelector(".main-container");
const image = document.getElementById("image");
let cropper;

// will modified later
document.getElementById('fileInput').addEventListener('change', function (event) {
  performCrop();
});

// document
//   .getElementById("perform-crop")
//   .addEventListener("click", function (event) {
//     event.preventDefault();
//     performCrop();
//   });

function performCrop() {
  submitFinalBtn.style.display = "block";
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  console.log(file);
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const imageElement = document.getElementById("image");
      imageElement.classList.remove("hidden");
      cropBtn.classList.remove("hidden");
      imageElement.src = e.target.result;
      document.getElementById("capturedImage").style.display = "none";

      if (cropper) {
        cropper.destroy();
      }

      cropper = new Cropper(image, {
        aspectRatio: 0,
      });

      document
        .getElementById("btn-crop")
        .addEventListener("click", function (event) {
          event.preventDefault();
          const croppedImage = cropper
            .getCroppedCanvas()
            .toDataURL("image/png");
          console.log("Cropped Image Data:", croppedImage);
          croppedContainer.style.display = "block";
          croppedOutput.src = croppedImage;
          cropper.destroy();
          console.log(croppedOutput.src);
          document.getElementById("show-the-img").style.display = "none";
          cropBtn.style.display = "none";

          const originalFileName = file.name.replace(/\.[^/.]+$/, "");
          const newFileName = `${originalFileName}_cropped.png`;

          const imageDataUrl = croppedImage;

          const blob = dataURLtoBlob(imageDataUrl);
          const newFile = new File([blob], newFileName);
          const fileList = new DataTransfer();
          fileList.items.add(newFile);
          fileInput.files = fileList.files;
        });
    };
    reader.readAsDataURL(file);
  }
}
