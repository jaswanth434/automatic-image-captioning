document.addEventListener("DOMContentLoaded", function () {
  // Handles image selection
  function imageSelected() {
    const fileInput = document.getElementById("fileInput");
    const uploadChangeButtons = document.getElementById("uploadChangeButtons");
    const chooseButton = document.getElementById("chooseButton");
    const fileNameTextArea = document.getElementById("fileName");

    let fileName = fileInput.files[0] ? fileInput.files[0].name : "";

    if (fileName.length > 25) {
      fileName =
        fileName.substring(0, 10) +
        "..." +
        fileName.substring(fileName.length - 6);
    }

    fileNameTextArea.textContent = fileName;
    chooseButton.classList.add("fadeSlideRight");

    setTimeout(function () {
      chooseButton.style.display = "none";
      uploadChangeButtons.style.display = "block";
      fileNameTextArea.style.display = "inline-block";
      fileNameTextArea.classList.add("brightSlideLeft");
      uploadChangeButtons.classList.add("brightSlideLeft");
    }, 500);
  }

  // Handles changing the selected image
  function changeImage() {
    const fileInput = document.getElementById("fileInput");
    const uploadChangeButtons = document.getElementById("uploadChangeButtons");
    const chooseButton = document.getElementById("chooseButton");
    const fileNameTextArea = document.getElementById("fileName");

    fileInput.value = "";
    uploadChangeButtons.classList.remove("brightSlideLeft");
    fileNameTextArea.classList.remove("brightSlideLeft");
    chooseButton.classList.remove("fadeSlideRight");

    setTimeout(function () {
      uploadChangeButtons.style.display = "none";
      fileNameTextArea.style.display = "none";
      chooseButton.style.display = "block";
      chooseButton.classList.add("brightSlideLeft");
    }, 500);
  }

  // Handles the upload of the image and fetches the caption
  function uploadImage() {
    const formData = new FormData(document.getElementById("uploadForm"));
    const image = document.getElementById("uploadedImage");
    const fileInput = document.querySelector("input[type=file]");

    if (fileInput.files && fileInput.files[0]) {
      image.src = URL.createObjectURL(fileInput.files[0]);
      image.style.display = "block";

      const loader = document.getElementById("loader");
      const responseContainer = document.getElementById("responseContainer");
      const responseTextElement = document.getElementById("responseText");

      loader.style.display = "inline-block";
      responseContainer.style.display = "none";
      responseTextElement.textContent = "";

      fetch("/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          loader.style.display = "none";
          responseContainer.style.display = "block";
          typeWriterEffect(data.message, responseTextElement);
        })
        .catch((error) => {
          console.error("Error:", error);
          loader.style.display = "none";
        });
    }
  }

  // Typewriter effect for displaying the response
  function typeWriterEffect(message, element) {
    let i = 0;
    function typeWriter() {
      if (i < message.length) {
        element.textContent += message.charAt(i);
        i++;
        let randomDelay =
          Math.random() < 0.5 ? Math.random() * 50 : Math.random() * 450;
        setTimeout(typeWriter, randomDelay);
      } else {
        element.classList.add("caption-response-complete");
      }
    }
    typeWriter();
  }

  // Event listeners
  document
    .getElementById("fileInput")
    .addEventListener("change", imageSelected);
  document
    .getElementById("chooseButton")
    .addEventListener("click", function () {
      document.getElementById("fileInput").click();
    });
  document
    .getElementById("uploadChangeButtons")
    .querySelectorAll("button")
    .forEach(function (button) {
      button.addEventListener(
        "click",
        button.id === "uploadChangeButtons" ? uploadImage : changeImage
      );
    });
});
