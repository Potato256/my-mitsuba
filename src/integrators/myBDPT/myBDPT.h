#include <mitsuba/render/scene.h>

#define MTS_OPENMP

MTS_NAMESPACE_BEGIN

struct BDPTVertex{
    Point3 pos;
    Normal n;
    Vector3 wi;
    Spectrum value;
    Point2 uv;
    const BSDF *bsdf;
    const Emitter *e;
    /* The local pdf of sampling this vertex */
    Float pdf;
    /* The local pdf of sampling last vertex inversely */
    Float pdfInverse;
    /* The pdf of sampling this vertex's position on light */
    Float pdfLight;
    /* Trace septh */
    int depth;

    std::string toString() const {
        std::ostringstream oss;
        oss << "pos:        "<< pos.toString() + "\n" 
            << "n:          "<< n.toString() + "\n"
            << "wi:         "<< wi.toString() + "\n"
            << "value:      "<< value.toString() + "\n"
            << "uv:         "<< uv.toString() + "\n"
            << "pdf:        "<<  pdf << "\n"
            << "pdfInverse: "<<  pdfInverse << "\n"
            << "pdfLight:   "<<  pdfLight << "\n";
        return oss.str();
    }    
};

/// wi point outwards
Float computePdfForward(Vector3& wi, BDPTVertex* mid, BDPTVertex* next) {
    Intersection its;
    its.p = mid->pos;
    its.shFrame = Frame(mid->n);
    its.uv = mid->uv;
    its.hasUVPartials = 0;
    Vector3 d = next->pos - mid->pos;
    Float distSquared = d.lengthSquared();
    d /= sqrt(distSquared);
    BSDFSamplingRecord bRec(its, its.toLocal(wi), its.toLocal(d), ERadiance);
    Float pdfBsdf = mid->bsdf->pdf(bRec);
    return pdfBsdf * absDot(next->n, d) / distSquared;
}

Float computePdfLightDir(BDPTVertex* l, BDPTVertex* e) {
    PositionSamplingRecord pRec(0.0f);
    pRec.n = l->n;
    Vector3 d = e->pos - l->pos;
    Float distSquared = d.lengthSquared();
    d /= sqrt(distSquared);
    DirectionSamplingRecord dRec(d);
    Float ans = l->e->pdfDirection(dRec, pRec) * absDot(e->n, d) / distSquared;
    return ans;        
}

bool evalContri(
    BDPTVertex* eyeEnd,
    BDPTVertex* lightEnd, 
    const Scene* scene, 
    Spectrum& contribution 
) {
    if (eyeEnd->value.max() <= 0 || lightEnd->value.max() <=0)
        return false;
    
    Vector d = lightEnd->pos - eyeEnd->pos;
    Float distSquared = d.lengthSquared();
    Float dist = std::sqrt(distSquared);
    d /= dist;
    Ray ray(eyeEnd->pos, d, Epsilon,
                dist*(1-ShadowEpsilon), 0.0f);
    
    if (scene->rayIntersect(ray))
        return false;
    Intersection its;
    its.p = eyeEnd->pos;
    its.shFrame = Frame(eyeEnd->n);
    its.uv = eyeEnd->uv;
    its.hasUVPartials = 0;
    
    /* Allocate a record for querying the BSDF */
    BSDFSamplingRecord bRec(its, its.toLocal(eyeEnd->wi), its.toLocal(d), ERadiance);
    Spectrum bsdfVal1 = eyeEnd->bsdf->eval(bRec);
    if (lightEnd->e != nullptr) {
        DirectionSamplingRecord dRec(-d);
        PositionSamplingRecord pRec(0.0f);
        pRec.n = lightEnd->n;
        contribution = eyeEnd->value * lightEnd->value * 
            lightEnd->e->evalDirection(dRec, pRec) * bsdfVal1 
            / distSquared;
    } else {
        its.p = lightEnd->pos;
        its.shFrame = Frame(lightEnd->n);
        its.uv = lightEnd->uv;
        BSDFSamplingRecord bRec(its, its.toLocal(lightEnd->wi), its.toLocal(-d),ERadiance);
        Spectrum bsdfVal2 = lightEnd->bsdf->eval(bRec);
        contribution = eyeEnd->value * lightEnd->value * bsdfVal1 * bsdfVal2
            / distSquared;
    }
    return true;
}

/**
 *  This function does a lot of useless work, so it's 
 *  only used to check correctness.
*/
Float computePathPdf(
    std::vector<BDPTVertex*> eyePath, 
    std::vector<BDPTVertex*> lightPath,
    int eyeEnd, 
    int lightEnd
)  {
    Float curPdf = lightPath[0]->pdfLight;
    for (int i = 0; i < eyeEnd; ++i)
        curPdf *= computePdfForward(eyePath[i]->wi, eyePath[i], eyePath[i+1]);
    if (lightEnd > 0) 
        curPdf *= computePdfLightDir(lightPath[0], lightPath[1]);
    for (int i = 1; i <lightEnd; ++i)
        curPdf *= computePdfForward(lightPath[i]->wi, lightPath[i], lightPath[i+1]);
    return curPdf;
}

/**
 *  This function does a lot of useless work, so it's 
 *  only used to check correctness.
*/
Float computePathMIS(
    std::vector<BDPTVertex*> eyePath, 
    std::vector<BDPTVertex*> lightPath,
    int eyeEnd, 
    int lightEnd,
    bool usePT
) {
    int numStrategy = eyeEnd + lightEnd + 1 + usePT;
    int curStrategy = eyeEnd;
    std::vector<BDPTVertex*> full; 
    for (int i = 0; i <= eyeEnd; ++i)
        full.push_back(eyePath[i]);
    for (int i = lightEnd; i >= 0; --i)
        full.push_back(lightPath[i]);
    Float nominator = 0;
    Float denominator = 0;
    for (int i = 0; i < numStrategy; ++i){
        std::vector<BDPTVertex*> e; 
        std::vector<BDPTVertex*> l;
        for (int j = 0; j <= i; ++j)
            e.push_back(full[j]);
        for (int j = numStrategy - 1; j > i; --j)
            l.push_back(full[j]);
        if (l.empty())
            l.push_back(full.back());
        Float pdf = computePathPdf(e, l, i, numStrategy-i-2);
        denominator += pdf;
        if (i == curStrategy)
            nominator += pdf;
    }
    return nominator / denominator;
}


MTS_NAMESPACE_END