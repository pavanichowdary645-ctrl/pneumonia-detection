let dropArea = document.getElementById("dropArea");
let fileInput = document.getElementById("fileInput");
let fileNameLabel = document.getElementById("fileName");

// Handle file selection
fileInput.addEventListener("change", function () {
    fileNameLabel.innerText = fileInput.files[0].name;
});

// Drag over highlight
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.background = "#eef6ff";
});

// Drag leave
dropArea.addEventListener("dragleave", () => {
    dropArea.style.background = "white";
});

// Drop file
dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.style.background = "white";

    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileNameLabel.innerText = e.dataTransfer.files[0].name;
    }
});
