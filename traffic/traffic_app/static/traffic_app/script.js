// document.addEventListener("DOMContentLoaded", function(){
//     const imageInput = document.getElementById('image-input');
//     const generateCaptionBtn = document.getElementById('generate-caption-btn');
//     const uploadedImage = document.getElementById('uploaded-image');
//     const generatedCaption = document.getElementById('generated-caption');

//     imageInput.addEventListener('change', (e) => {
//         const file = imageInput.files[0];
//         const reader = new FileReader();
//         reader.onload = (e) => {
//             uploadedImage.src = e.target.result;
//         };
//         reader.readAsDataURL(file);
//     });

//     generateCaptionBtn.addEventListener('click', () => {
//         // TO DO: Implement caption generation logic here
//         // For demonstration purposes, I'll just generate a random caption
//         const captions = [
//             'A beautiful sunset',
//             'A happy dog playing',
//             'A delicious meal',
//             'A stunning landscape',
//         ];
//         const randomCaption = captions[Math.floor(Math.random() * captions.length)];
//         generatedCaption.textContent = randomCaption;
//         });
// })

