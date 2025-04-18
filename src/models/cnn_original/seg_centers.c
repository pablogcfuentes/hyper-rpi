#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAXDATSIZE 2147483648u

// #define SEG "/mnt/media/images/seg_salinas.raw"
// #define CENTER "/mnt/media/images/seg_salinas_centers.raw"

// #define SEG "/mnt/media/images/seg_paviau.raw"
// #define CENTER "/mnt/media/images/seg_paviau_centers.raw"

#define SEG "/mnt/media/images/seg_oitaven_wp.raw"
#define CENTER "/mnt/media/images/seg_oitaven_wp_centers.raw"

// #define SEG "/mnt/media/images/seg_ermidas.raw"
// #define CENTER "/mnt/media/images/seg_ermidas_centers.raw"

unsigned int *read_seg_raw(const char *fichero, int *H1, int *V1)
{  FILE *fp; int *datos, H=0, V=0; size_t a;
   fp=fopen(fichero,"rb");
   if(fp==NULL) { printf("# Cannot open file %s\n",fichero); return(NULL); }
   else printf("\n* Open DATASET %s\n",fichero);
   printf("  File type: raw\n");
   a=fread(&H,4,1,fp); a=fread(&V,4,1,fp);
   if((H<=0)||(V<=0)) { printf("# Incorrect dimensions\n"); return(NULL); }
   else if((size_t)H*V>=MAXDATSIZE) { printf("# Dataset too large\n"); return(NULL); } 
   datos=(unsigned int *)malloc((size_t)H*V*sizeof(int));
   if(datos==NULL) { printf("Out of memory\n"); return(NULL); }
   a=fread(datos,4,H*V,fp); 
   if(a!=(size_t)H*V) { printf("# Read failure %s\n",fichero); return(NULL); }
   printf("  read %lu bytes\n",(size_t)4*a);
   fclose(fp); *H1=H; *V1=V; return(datos);
}

unsigned int *seg_centers(int *seg, int H, int V, int *nseg1)
{  int i, j, u, nseg=0, N=H*V;
   for(i=0;i<N;i++) if(seg[i]>nseg) nseg=seg[i];
   nseg+=1;
   unsigned int *center=(unsigned int *)malloc((size_t)N*sizeof(int));
   unsigned int *xmin=(unsigned int *)malloc((size_t)N*sizeof(int));
   unsigned int *ymin=(unsigned int *)malloc((size_t)N*sizeof(int));
   unsigned int *xmax=(unsigned int *)calloc((size_t)N,sizeof(int));
   unsigned int *ymax=(unsigned int *)calloc((size_t)N,sizeof(int));
   if(center==NULL || xmin==NULL  || ymin==NULL || xmax==NULL || xmax==NULL) 
   { printf("Out of memory\n"); exit(0); }
   for(i=0;i<N;i++) xmin[i]=N;
   for(i=0;i<N;i++) ymin[i]=N;
   for(i=0;i<V;i++) for(j=0;j<H;j++)
   {  u=seg[i*H+j];
      if(j<xmin[u]) xmin[u]=j; if(j>xmax[u]) xmax[u]=j;
      if(i<ymin[u]) ymin[u]=i; if(i>ymax[u]) ymax[u]=i; }
   for(u=0;u<nseg;u++) 
   {  j=(xmax[u]+xmin[u])/2; i=(ymax[u]+ymin[u])/2;
      center[u]=i*H+j; }
   free(xmin); free(xmax); free(ymin); free(ymax); 
   *nseg1=nseg; return(center);
}

void save_centers_raw_old(const char *fichero, unsigned int *center, int H, int V, int nseg)
{  FILE *fp; int p; size_t a;
   fp=fopen(fichero,"w");
   if(fp==NULL) { printf("# Cannot open file %s\n",fichero); return; }
   else printf("* Saving CENTERS %s\n",fichero);
   p=1; a=fwrite(&p,4,1,fp);
   p=nseg; a=fwrite(&p,4,1,fp);
   p=1; a=fwrite(&p,4,1,fp);
   a=fwrite(center,4,nseg,fp);
   if(a!=nseg) printf("# Write failure %s\n",fichero);
   else printf("  written %d segments\n",nseg);
}

void save_raw_bare(const char *fichero, int *img, int H, int V, int B)
{  FILE *fp; int p; size_t a;
   fp=fopen(fichero,"wb");
   if(fp==NULL) { perror("No puedo abrir fichero %s\n"); return; }
   p=B; a=fwrite(&p,4,1,fp);
   p=H; a=fwrite(&p,4,1,fp);
   p=V; a=fwrite(&p,4,1,fp);
   a=fwrite(img,(size_t)H*V*B*4,1,fp);
   if(a==0) printf("Error en save_raw_bare\n");
   printf("Save file %s\n",fichero);
   fclose(fp);
}

int main()
{  int H, V, nseg;
   unsigned int *seg=read_seg_raw(SEG,&H,&V);
   unsigned int *center=seg_centers(seg,H,V,&nseg);
   // save_centers_raw(CENTER,center,H,V,nseg);
   save_raw_bare(CENTER,center,nseg,1,1);
}
   
