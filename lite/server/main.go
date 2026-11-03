// bernina-apps: serve the static browser build of the Bernina apps from the
// "site" folder next to this program and open it in the default browser.
//
// The apps run entirely in the browser (Python compiled to WebAssembly), but a
// browser will not load them from file:// - they need to come from a web
// server, even a local one. This is that server: no Python, no installation,
// one file per operating system. It listens on 127.0.0.1 only, so nothing is
// reachable from other machines.
package main

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	exe, err := os.Executable()
	if err != nil {
		fail(err)
	}
	site := filepath.Join(filepath.Dir(exe), "site")
	if _, err := os.Stat(filepath.Join(site, "index.html")); err != nil {
		fail(fmt.Errorf("no site folder next to this program (%s)", site))
	}
	ln, err := listen()
	if err != nil {
		fail(err)
	}
	url := fmt.Sprintf("http://%s/voici/render/diffraction_reflections.html", ln.Addr())
	fmt.Println("Bernina apps running at")
	fmt.Println("   ", url)
	fmt.Printf("    (all apps: http://%s/)\n", ln.Addr())
	fmt.Println("Close this window to stop.")
	openBrowser(url)
	if err := http.Serve(ln, noCache(http.FileServer(http.Dir(site)))); err != nil {
		fail(err)
	}
}

// listen takes the first free port from 8870 on, so a second copy (or another
// program) on the usual port does not stop it from starting.
func listen() (net.Listener, error) {
	var last error
	for port := 8870; port < 8900; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err == nil {
			return ln, nil
		}
		last = err
	}
	return nil, last
}

// noCache makes the browser re-check files, so an updated site is picked up
// without clearing the cache (the large kernel packages still come from the
// browser cache when unchanged: revalidation is cheap).
func noCache(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache")
		h.ServeHTTP(w, r)
	})
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	case "darwin":
		cmd = exec.Command("open", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	_ = cmd.Start() // no browser found: the URL is printed above
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	fmt.Fprintln(os.Stderr, "press Enter to close")
	fmt.Scanln()
	os.Exit(1)
}
