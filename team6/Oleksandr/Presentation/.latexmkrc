$out_dir = 'out';
$pdf_mode = 1;        # 1=pdflatex, 4=lualatex, 5=xelatex
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
$lualatex = 'lualatex -interaction=nonstopmode -halt-on-error %O %S';
$xelatex  = 'xelatex -interaction=nonstopmode -halt-on-error %O %S';

# Copy PDFs from $out_dir back to source dir after a successful build
END {
    if (-d $out_dir) {
        my @pdfs = glob("$out_dir/*.pdf");
        foreach my $pdf (@pdfs) {
            my $base = $pdf;
            $base =~ s|.*/||;
            system("cp '$pdf' '$base'");
        }
    }
}
