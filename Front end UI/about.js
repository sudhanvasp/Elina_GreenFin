// Add any page-specific interactive or animation JavaScript here
// Example: Animate the about-container to fade in on page load
window.addEventListener('DOMContentLoaded', (event) => {
    const aboutContainer = document.querySelector('.about-container');
    aboutContainer.style.opacity = 0;
    setTimeout(() => {
      aboutContainer.style.opacity = 1;
      aboutContainer.style.transition = 'opacity 2s ease-out';
    }, 100);
  });
  