var fos = fos || {};

// Show an element
fos.show = function(e) {
  // e.preventDefault();
	e.classList.add('js-is-visible');
  e.classList.remove('js-is-hidden');
};

// Hide an element
fos.hide = function(e) {
  // e.preventDefault();
  e.classList.add('js-is-hidden');
	e.classList.remove('js-is-visible');
};

fos.clickListen = function(e) {
  // Detect all clicks on the document
  document.addEventListener("click", function(e) {

    // If user clicks inside the element, do nothing
    if (e.target.closest(".box")) return;

    // If user clicks outside the element, hide it
    var container = document.querySelector('.box-container');
    container.classList.add("js-is-hidden");
    container.classList.remove("js-is-visible");
  });
};

fos.socialMenu = function(e) {
  // querySelector returns the first element it finds with the correct selector
  // addEventListener is roughly analogous to $.on()
  document.querySelector('#social-links-toggle').addEventListener('click', function(e) {
    e.preventDefault();
    var container = document.querySelector('#social-links-container');
    if (container.classList.contains('js-is-visible'))
      fos.hide(container);
    else
      fos.show(container);
  });
};

fos.jsCheck = function(e) {
  var bodyClass = document.querySelector('html').classList;
  bodyClass.remove('no-js');
  bodyClass.add('js');
};


// start everything
// this isn't in a doc.ready - loaded at the bottom of the page so the DOM is already ready
fos.jsCheck();
fos.socialMenu();
fos.clickListen();

SocialShareKit.init();
